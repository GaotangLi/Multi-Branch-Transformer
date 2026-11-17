"""The explicit MBT computation graph; no hooks or process-global model state."""
from __future__ import annotations

import math
from functools import partial

import torch
from torch import nn

from .config import ModelConfig
from .modules import BranchStack, CheckpointModule, ConcatFusion, JointKVProjection, WidthBridge


class MBTDecoder(CheckpointModule):
    """Shared prelude → independent branches → fusion → shared coda.

    ``layer_factory(index, shape, shared_kv)`` constructs one native block.
    Keeping construction explicit also allows CPU tests to exercise this exact
    graph with small diagnostic blocks, without importing CUDA dependencies.
    """

    def __init__(self, model_config: ModelConfig, layer_factory, final_norm: nn.Module, *,
                 config=None, device=None, dtype=None, joint_kv_factory=None, tp_group=None):
        super().__init__()
        self.model_config = model_config
        self.config = config
        self.tp_group = tp_group
        self.input_tensor = None
        self.kv_mode = model_config.kv_mode
        self.prelude = nn.ModuleList()
        self.branches = nn.ModuleList()
        self.coda = nn.ModuleList()
        physical_index = 0
        prelude_depth = model_config.prelude if model_config.branches else model_config.physical_layers
        for _ in range(prelude_depth):
            self.prelude.append(layer_factory(physical_index, model_config.outer_shape, False))
            physical_index += 1
        for branch_depth in model_config.branches:
            blocks = []
            for _ in range(branch_depth):
                blocks.append(layer_factory(physical_index, model_config.branch_shape, self.kv_mode == "joint"))
                physical_index += 1
            self.branches.append(BranchStack(blocks, tp_group=tp_group))
        for _ in range(model_config.coda or 0):
            self.coda.append(layer_factory(physical_index, model_config.outer_shape, False))
            physical_index += 1
        self.final_layernorm = final_norm

        base_init = getattr(config, "init_method", None) or partial(nn.init.normal_, std=.02)
        output_init = getattr(config, "output_layer_init_method", None) or partial(
            nn.init.normal_, std=.02/math.sqrt(2*model_config.physical_layers)
        )
        outer_width = model_config.outer_shape[0]
        branch_width = model_config.branch_shape[0]
        self.width_in = WidthBridge(outer_width, branch_width, initializer=base_init,
            device=device, dtype=dtype, tp_group=tp_group) if model_config.uses_bridge else None
        self.fusion = ConcatFusion(branch_width, outer_width, len(self.branches),
            initializer=output_init, device=device, dtype=dtype, tp_group=tp_group) if self.branches else None
        self.joint_kv = nn.ModuleList()
        if self.kv_mode == "joint":
            for depth in range(model_config.branches[0]):
                producer = (joint_kv_factory(depth) if joint_kv_factory is not None else
                    JointKVProjection(branch_width, len(self.branches), initializer=base_init,
                        device=device, dtype=dtype, tp_group=tp_group))
                self.joint_kv.append(producer)

    @property
    def layers(self):
        """Physical block order for diagnostics; modules are registered once."""
        return tuple(self.prelude) + tuple(
            block for branch in self.branches for block in branch.blocks
        ) + tuple(self.coda)

    def get_layer_execution_order(self):
        if self.kv_mode != "joint":
            return tuple(range(len(self.layers)))
        prelude_depth = len(self.prelude)
        branch_depth = len(self.branches[0].blocks)
        branch_indices = tuple(prelude_depth + branch*branch_depth + depth
            for depth in range(branch_depth) for branch in range(len(self.branches)))
        coda_start = prelude_depth + len(self.branches)*branch_depth
        return tuple(range(prelude_depth)) + branch_indices + tuple(range(coda_start, len(self.layers)))

    def set_input_tensor(self, tensor):
        self.input_tensor = tensor

    def forward(self, hidden: torch.Tensor, attention_mask=None, rotary_pos_emb=None):
        for block in self.prelude:
            hidden = block(hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)

        if self.branches:
            branch_input = self.width_in(hidden) if self.width_in is not None else hidden
            states = tuple(branch_input.clone() for _ in self.branches)

            if self.kv_mode == "independent":
                states = tuple(branch(state, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
                    for branch, state in zip(self.branches, states))
            else:
                for depth, kv_projection in enumerate(self.joint_kv):
                    # Form the shared KV from ALL inputs at this depth before
                    # updating any branch. Gradients flow to every contributor.
                    normalized_inputs = tuple(branch.blocks[depth].normalized_attention_input(state)
                        for branch, state in zip(self.branches, states))
                    shared_kv = kv_projection(*normalized_inputs)
                    states = tuple(branch.blocks[depth](state, attention_mask=attention_mask,
                        rotary_pos_emb=rotary_pos_emb, shared_kv=shared_kv)
                        for branch, state in zip(self.branches, states))

            hidden = self.fusion(*states)

        for block in self.coda:
            hidden = block(hidden, attention_mask=attention_mask, rotary_pos_emb=rotary_pos_emb)
        return self.final_layernorm(hidden)
