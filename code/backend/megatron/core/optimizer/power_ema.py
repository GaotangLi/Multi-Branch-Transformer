# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Power-function EMA of model weights (EDM2, Karras et al. 2024).

Maintains one or more exponential-moving-average copies of the trained weights
using the *power-function* profile, in which the averaging window grows with the
step count instead of using a fixed decay:

    beta_t = (t / (t + 1)) ** (1 + gamma)          (t = number of updates so far)
    EMA_t  = beta_t * EMA_{t-1} + (1 - beta_t) * theta_t

Larger ``gamma`` -> faster-forgetting (narrower) average concentrated near the
end of training. The first update seeds EMA_0 = theta_0 (equivalent to beta = 0).

Memory / compute notes
----------------------
The EMA is kept over the optimizer's fp32 *master* parameters
(``optimizer.get_parameters()``). Under the Megatron distributed optimizer those
master tensors are already the per-rank shards, so each rank stores only its
1/DP slice per gamma -> total extra memory ``num_gammas * (fp32 params) / DP``.
Updates are a single fused ``_foreach_mul_`` + ``_foreach_add_`` per gamma.

To evaluate or checkpoint an EMA snapshot, :meth:`apply_to_model` temporarily
writes the EMA shards into the master params and reuses the optimizer's existing
main->model scatter, so no bespoke all-gather is required.
"""

import contextlib
from typing import Dict, List, Sequence

import torch


class PowerEMA:
    """Power-function EMA over an optimizer's fp32 master parameters.

    Args:
        optimizer: a ``MegatronOptimizer`` whose ``get_parameters()`` returns the
            fp32 master tensors updated by the inner ``optimizer.step()``.
        gammas: one or more power-EMA gamma values (e.g. ``[12.5, 66.7]``).
    """

    def __init__(self, optimizer, gammas: Sequence[float]):
        assert len(gammas) > 0, "PowerEMA needs at least one gamma"
        self.optimizer = optimizer
        self.gammas: List[float] = [float(g) for g in gammas]
        # Only non-empty master shards participate (a rank may own no slice of a
        # given param). Order is fixed and mirrors get_parameters().
        self._params: List[torch.Tensor] = [
            p for p in optimizer.get_parameters() if p is not None and p.numel() > 0
        ]
        self.shadow: Dict[float, List[torch.Tensor]] = {}
        self.num_updates: int = 0

    @torch.no_grad()
    def update(self) -> None:
        """Fold the current master params into every EMA. Call after a step."""
        if not self.shadow:
            # Seed EMA_0 = theta_0 (beta = 0 on the first update).
            for gamma in self.gammas:
                self.shadow[gamma] = [p.detach().clone().float() for p in self._params]
            self.num_updates = 1
            return

        t = self.num_updates
        for gamma in self.gammas:
            beta = (t / (t + 1.0)) ** (1.0 + gamma)
            ema = self.shadow[gamma]
            torch._foreach_mul_(ema, beta)
            torch._foreach_add_(ema, self._params, alpha=1.0 - beta)
        self.num_updates += 1

    @contextlib.contextmanager
    def apply_to_model(self, gamma: float):
        """Temporarily load the ``gamma`` EMA into the live model, then restore.

        Backs up the fp32 master shards, copies the EMA shards in, scatters them
        to the model params (reusing the optimizer's main->model copy), yields,
        then restores the originals. Use around eval / checkpoint export.
        """
        assert gamma in self.shadow, f"No EMA for gamma={gamma}"
        backup = [p.detach().clone() for p in self._params]
        try:
            torch._foreach_copy_(self._params, self.shadow[gamma])
            self._scatter_master_to_model()
            yield
        finally:
            torch._foreach_copy_(self._params, backup)
            self._scatter_master_to_model()

    @contextlib.contextmanager
    def model_ema_for_save(self, gamma: float):
        """Put the ``gamma`` EMA into the *model* params while leaving the fp32
        master untouched, so a checkpoint saved inside this context holds EMA
        weights in the model while the optimizer state stays raw (resumable).

        On exit the model is re-synced from the (raw) master.
        """
        assert gamma in self.shadow, f"No EMA for gamma={gamma}"
        backup_master = [p.detach().clone() for p in self._params]
        try:
            torch._foreach_copy_(self._params, self.shadow[gamma])
            self._scatter_master_to_model()            # model <- EMA
            torch._foreach_copy_(self._params, backup_master)  # master <- raw (model stays EMA)
            yield
        finally:
            self._scatter_master_to_model()            # model <- raw

    def restore_model_from_master(self) -> None:
        """Re-sync the live model params from the (raw) fp32 master. Use after
        loading a checkpoint whose model weights are EMA, so training resumes from
        the raw weights held in the optimizer master state."""
        self._scatter_master_to_model()

    def _scatter_master_to_model(self) -> None:
        # Mixed-precision / distributed optimizers keep sharded fp32 masters and
        # must scatter them onto the (bf16) model params. Plain fp32 optimizers
        # return the model params directly from get_parameters(), so the in-place
        # copy above already updated the model.
        copy_fn = getattr(self.optimizer, "_copy_main_params_to_model_params", None)
        if copy_fn is None:
            return
        copy_fn()
        # _copy_main_params_to_model_params only fills each rank's local shard region
        # of the param buffer; force the DDP all-gather so the *full* model params are
        # materialized. This matters for saving (no forward pass triggers the gather).
        for model_chunk in getattr(self.optimizer, "model_chunks", []):
            if hasattr(model_chunk, "start_param_sync"):
                model_chunk.start_param_sync(force_sync=True)

    def state_dict(self) -> dict:
        """Per-rank EMA state (CPU tensors). Assumes a fixed parallel layout."""
        return {
            "num_updates": self.num_updates,
            "gammas": self.gammas,
            "shadow": {g: [t.detach().cpu() for t in ts] for g, ts in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.num_updates = state_dict["num_updates"]
        loaded_gammas = state_dict.get("gammas", self.gammas)
        assert list(loaded_gammas) == self.gammas, (
            f"PowerEMA gammas changed on resume: checkpoint {loaded_gammas} vs {self.gammas}"
        )
        self.shadow = {}
        for g, ts in state_dict["shadow"].items():
            g = float(g)
            self.shadow[g] = [
                t.to(device=p.device, dtype=torch.float32)
                for t, p in zip(ts, self._params)
            ]
