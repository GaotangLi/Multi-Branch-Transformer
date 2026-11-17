# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MuLoCo outer optimization over Megatron optimizer-owned FP32 shards.

This module intentionally does not update BF16 model parameters. After a
successful outer step, the caller must copy/scatter the updated FP32 master
parameters into the model and then call :meth:`mark_model_params_refreshed`.
For checkpoint resume, construct the controller, call :meth:`load_state_dict`,
then call the idempotent :meth:`start` before training.

TorchFT treats each independent Megatron worker world as one replica. Every rank
in a replica owns a Manager configured with its replica-local ``rank`` and common ``world_size``;
the reconfigurable process group then connects corresponding ranks across
replicas. Consequently, ``optimizer.get_parameters()`` must return the same
ordered shard layout on corresponding ranks in every replica.
"""

from __future__ import annotations

import hashlib
import logging
import socket
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from megatron.core import communication_probes

try:
    import torchft as _torchft
except ImportError:
    _torchft = None


logger = logging.getLogger(__name__)


class TorchFTUnavailableError(RuntimeError):
    """Raised when MuLoCo is asked to create TorchFT objects without TorchFT."""


@dataclass(frozen=True)
class TorchFTReplicaConfig:
    """Configuration for one TorchFT replica-local Manager group.

    ``rank`` and ``world_size`` describe ranks *inside one replica*, not global
    job ranks. ``store_addr`` and ``store_port`` must identify a store dedicated
    to this replica because TorchFT Manager store keys are replica-local.
    """

    min_replica_size: int
    rank: int
    world_size: int
    store_addr: str
    store_port: int
    lighthouse_addr: Optional[str] = None
    replica_id: Optional[str] = None
    manager_port: Optional[int] = None
    hostname: str = field(default_factory=socket.gethostname)
    timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    quorum_timeout: timedelta = field(default_factory=lambda: timedelta(hours=1))
    connect_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=60))
    init_sync: bool = False

    def __post_init__(self) -> None:
        if self.min_replica_size < 1:
            raise ValueError("min_replica_size must be at least 1")
        if self.world_size < 1:
            raise ValueError("world_size must be at least 1")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.store_port <= 0:
            raise ValueError("store_port must be positive")


MasterParamsUpdatedCallback = Callable[["MuLoCoController", bool], None]


class MuLoCoController:
    """Periodically apply a TorchFT-averaged outer SGD update.

    The pseudo-gradient is ``local_fp32_param - anchor``. TorchFT averages that
    delta across replicas, and outer SGD applies it with the configured
    ``outer_maximize`` direction. The committed anchors are copied into the
    optimizer-owned FP32 parameters, but model parameters are not refreshed.

    Args:
        optimizer: Megatron optimizer exposing ``get_parameters()``. All
            non-empty returned tensors must be FP32 owned shards.
        sync_every: Number of successful inner optimizer updates per outer step.
        outer_lr: Outer SGD learning rate.
        outer_momentum: Outer SGD momentum.
        outer_nesterov: Whether outer SGD uses Nesterov momentum.
        outer_maximize: Whether outer SGD maximizes the local-minus-anchor
            delta. When false, SGD moves opposite that delta.
        manager: Optional prebuilt Manager-compatible object, primarily for
            embedding or tests. When omitted, ``torchft_config`` is required.
        torchft_config: Settings used to construct TorchFT ProcessGroupNCCL and
            Manager. Ignored when ``manager`` is supplied.
        verify_shard_signatures: Verify ordered shapes/dtypes across replicas
            before the first delta synchronization.
        on_master_params_updated: Optional callback invoked after owned FP32
            parameters are changed. Its bool argument reports whether the outer
            update committed; false denotes rollback after a failed quorum.
        strict_commit: Raise after restoring anchors when TorchFT does not
            commit. This is the safe default for fixed-K training because the
            consumed local data window cannot be replayed here.
        manager_state_key: Registration key used with an injected Manager.
    """

    _STATE_VERSION = 1

    def __init__(
        self,
        optimizer: Any,
        sync_every: int,
        outer_lr: float,
        outer_momentum: float = 0.0,
        outer_nesterov: bool = False,
        outer_maximize: bool = True,
        *,
        manager: Optional[Any] = None,
        torchft_config: Optional[TorchFTReplicaConfig] = None,
        verify_shard_signatures: bool = True,
        on_master_params_updated: Optional[MasterParamsUpdatedCallback] = None,
        strict_commit: bool = True,
        manager_state_key: str = "muloco",
    ) -> None:
        if sync_every < 1:
            raise ValueError("sync_every must be at least 1")
        if outer_lr < 0.0:
            raise ValueError("outer_lr must be non-negative")
        if outer_momentum < 0.0:
            raise ValueError("outer_momentum must be non-negative")
        if outer_nesterov and outer_momentum <= 0.0:
            raise ValueError("outer_nesterov requires positive outer_momentum")
        if manager is not None and torchft_config is not None:
            raise ValueError("pass either manager or torchft_config, not both")

        get_parameters = getattr(optimizer, "get_parameters", None)
        if not callable(get_parameters):
            raise TypeError("optimizer must expose a callable get_parameters()")

        parameters = []
        for index, param in enumerate(get_parameters()):
            if param is None:
                continue
            if not isinstance(param, torch.Tensor):
                raise TypeError(f"owned parameter {index} is not a torch.Tensor")
            if param.numel() > 0:
                parameters.append(param)
        if not parameters:
            raise ValueError(
                "optimizer.get_parameters() returned no non-empty parameters"
            )
        if len({id(param) for param in parameters}) != len(parameters):
            raise ValueError("optimizer.get_parameters() returned duplicate tensors")
        for index, param in enumerate(parameters):
            if param.dtype != torch.float32:
                raise TypeError(
                    f"owned parameter {index} must be FP32, found dtype={param.dtype}"
                )

        self.optimizer = optimizer
        self.sync_every = int(sync_every)
        self.outer_maximize = bool(outer_maximize)
        self._params: tuple[torch.Tensor, ...] = tuple(parameters)
        self._anchors: tuple[torch.nn.Parameter, ...] = tuple(
            torch.nn.Parameter(param.detach().clone()) for param in self._params
        )
        self._outer_optimizer = torch.optim.SGD(
            self._anchors,
            lr=float(outer_lr),
            momentum=float(outer_momentum),
            nesterov=bool(outer_nesterov),
            maximize=self.outer_maximize,
        )

        self._local_step = 0
        self._total_inner_steps = 0
        self._outer_steps = 0
        self._verify_shard_signatures = bool(verify_shard_signatures)
        self._signature_verified = False
        self._shard_signature = self._compute_shard_signature(self._params)
        self._on_master_params_updated = on_master_params_updated
        self._model_refresh_required = False
        self._strict_commit = bool(strict_commit)
        self._started = False
        self._closed = False
        self._manager: Optional[Any] = None
        self._process_group: Optional[Any] = None
        self._owns_manager = manager is None

        if manager is None:
            if torchft_config is None:
                raise ValueError(
                    "torchft_config is required when manager is not supplied"
                )
            self._process_group, self._manager = self._create_torchft_manager(
                torchft_config
            )
        else:
            self._manager = manager
            register = getattr(manager, "register_state_dict_fn", None)
            if callable(register):
                register(
                    manager_state_key,
                    self._load_controller_state_dict,
                    self._controller_state_dict,
                )

    @property
    def local_step(self) -> int:
        """Successful inner updates accumulated since the last sync attempt."""

        return self._local_step

    @property
    def total_inner_steps(self) -> int:
        """Total successful inner updates observed by this controller."""

        return self._total_inner_steps

    @property
    def outer_steps(self) -> int:
        """Number of committed outer updates."""

        return self._outer_steps

    @property
    def model_refresh_required(self) -> bool:
        """Whether FP32 masters changed since the caller last refreshed the model."""

        return self._model_refresh_required

    @property
    def anchors(self) -> Sequence[torch.Tensor]:
        """Read-only sequence of current outer anchor tensors."""

        return self._anchors

    def mark_model_params_refreshed(self) -> None:
        """Acknowledge that the caller propagated FP32 masters to model params."""

        self._model_refresh_required = False

    def start(self) -> None:
        """Start the first H-window quorum, once.

        Call this after loading controller/Manager checkpoint state. Collective
        methods also call it lazily so a fresh run cannot use an absent quorum.
        """

        self._ensure_open()
        if self._started:
            return
        assert self._manager is not None
        started_at = time.perf_counter()
        with torch.profiler.record_function("muloco::start_quorum"):
            self._manager.start_quorum()
        communication_probes.increment(
            "muloco_quorum_host_seconds", time.perf_counter() - started_at
        )
        communication_probes.increment("muloco_quorum_calls")
        self._started = True

    @torch.no_grad()
    def average_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Average ``tensor`` in place across the active TorchFT replica quorum.

        This is intended for auxiliary per-inner-step state such as FP32 router
        expert counts. It deliberately shares the H-window quorum used by the
        next outer synchronization and does not call ``should_commit``. Every
        corresponding rank must call auxiliary collectives in the same order.
        """

        self._ensure_open()
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("tensor must be a torch.Tensor")
        if not torch.is_floating_point(tensor):
            raise TypeError("average_tensor requires a floating-point tensor")
        self.start()
        assert self._manager is not None
        communication_probes.record_collective(
            "muloco_aux_allreduce", tensor.numel() * tensor.element_size()
        )
        lifetime_start = communication_probes.start_cuda_interval()
        with torch.profiler.record_function("muloco::aux_allreduce"):
            wait_start = communication_probes.start_cuda_interval()
            work = self._manager.allreduce(tensor)
            work.wait()
            communication_probes.finish_cuda_interval(
                "muloco_aux_allreduce_exposed", wait_start
            )
        communication_probes.finish_cuda_interval(
            "muloco_aux_allreduce_lifetime", lifetime_start
        )
        errored = getattr(self._manager, "errored", None)
        if callable(errored) and errored() is not None:
            raise RuntimeError("TorchFT failed while averaging MuLoCo auxiliary state")
        return tensor

    @torch.no_grad()
    def average_initial_parameters(self) -> None:
        """Average a fresh run's FP32 shards and reset all outer anchors.

        This makes identical initialization an enforced invariant rather than
        an assumption. It must run before any inner or outer optimizer update.
        """

        self._ensure_open()
        if self._local_step != 0 or self._total_inner_steps != 0 or self._outer_steps != 0:
            raise RuntimeError(
                "initial MuLoCo parameters can only be averaged before training starts"
            )
        for param, anchor in zip(self._params, self._anchors):
            self.average_tensor(param)
            anchor.copy_(param)
        self._model_refresh_required = True

    def after_inner_step(self, update_successful: bool = True) -> bool:
        """Record one inner step and synchronize on every ``sync_every`` successes.

        Returns true only when this call committed an outer synchronization.
        Failed/skipped inner optimizer updates do not advance the cadence.
        """

        self._ensure_open()
        if not update_successful:
            return False

        self._local_step += 1
        self._total_inner_steps += 1
        if self._local_step < self.sync_every:
            return False
        return self.synchronize()

    @torch.no_grad()
    def synchronize(self) -> bool:
        """Average local deltas and attempt one outer update immediately.

        A failed TorchFT commit rolls owned FP32 parameters back to the previous
        anchor. Either outcome resets the local-step cadence.
        """

        self._ensure_open()
        self.start()
        assert self._manager is not None
        communication_probes.increment("muloco_outer_syncs")
        outer_start = communication_probes.start_cuda_interval()

        if self._verify_shard_signatures and not self._signature_verified:
            with torch.profiler.record_function("muloco::outer/verify_shards"):
                self._verify_corresponding_shards()

        phase_start = communication_probes.start_cuda_interval()
        with torch.profiler.record_function("muloco::outer/build_deltas"):
            deltas = [
                param.detach().clone().sub_(anchor)
                for param, anchor in zip(self._params, self._anchors)
            ]
        communication_probes.finish_cuda_interval(
            "muloco_delta_build", phase_start
        )

        communication_probes.record_collective(
            "muloco_outer_allreduce",
            sum(delta.numel() * delta.element_size() for delta in deltas),
            calls=len(deltas),
        )
        phase_start = communication_probes.start_cuda_interval()
        with torch.profiler.record_function("muloco::outer/allreduce"):
            wait_start = communication_probes.start_cuda_interval()
            works = [self._manager.allreduce(delta) for delta in deltas]
            for work in works:
                work.wait()
            communication_probes.finish_cuda_interval(
                "muloco_outer_allreduce_exposed", wait_start
            )
        communication_probes.finish_cuda_interval(
            "muloco_outer_allreduce_lifetime", phase_start
        )

        phase_started_at = time.perf_counter()
        with torch.profiler.record_function("muloco::outer/should_commit"):
            committed = bool(self._manager.should_commit())
        communication_probes.increment(
            "muloco_commit_host_seconds", time.perf_counter() - phase_started_at
        )
        communication_probes.increment("muloco_commit_calls")
        self._started = False
        if committed:
            phase_start = communication_probes.start_cuda_interval()
            with torch.profiler.record_function("muloco::outer/optimizer"):
                for anchor, delta in zip(self._anchors, deltas):
                    anchor.grad = delta
                self._outer_optimizer.step()
                self._outer_optimizer.zero_grad(set_to_none=True)
            communication_probes.finish_cuda_interval(
                "muloco_outer_optimizer", phase_start
            )
            self._outer_steps += 1
            communication_probes.increment("muloco_outer_commits")
        else:
            communication_probes.increment("muloco_outer_commit_failures")

        # On failure, discard the local window. On success, install the new
        # outer anchors. Both cases require a later FP32-master -> model refresh.
        phase_start = communication_probes.start_cuda_interval()
        with torch.profiler.record_function("muloco::outer/install_anchors"):
            for param, anchor in zip(self._params, self._anchors):
                param.copy_(anchor)
        communication_probes.finish_cuda_interval(
            "muloco_anchor_install", phase_start
        )
        self._local_step = 0
        self._model_refresh_required = True
        if self._on_master_params_updated is not None:
            self._on_master_params_updated(self, committed)
        if not committed and self._strict_commit:
            raise RuntimeError(
                "TorchFT did not commit the MuLoCo H-step window; owned FP32 "
                "parameters were restored to the previous anchor"
            )
        # The next quorum covers all auxiliary averages and the delta
        # synchronization in the next H-step window.
        self.start()
        communication_probes.finish_cuda_interval("muloco_outer_sync", outer_start)
        return committed

    def state_dict(self) -> dict[str, Any]:
        """Return controller and TorchFT Manager checkpoint state."""

        state = self._controller_state_dict()
        manager_state_dict = getattr(self._manager, "state_dict", None)
        if callable(manager_state_dict):
            state["manager"] = manager_state_dict()
        return state

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Restore controller state; live/inner-optimizer state is caller-owned."""

        if self._started:
            raise RuntimeError("load MuLoCo state before start() establishes a quorum")
        self._load_controller_state_dict(state_dict)
        manager_state = state_dict.get("manager")
        manager_load_state_dict = getattr(self._manager, "load_state_dict", None)
        if manager_state is not None and callable(manager_load_state_dict):
            manager_load_state_dict(manager_state)

    def close(self) -> None:
        """Release TorchFT resources created by this controller."""

        if self._closed:
            return
        self._closed = True
        self._started = False
        if not self._owns_manager:
            return

        shutdown = getattr(self._manager, "shutdown", None)
        if callable(shutdown):
            shutdown()
        pg_shutdown = getattr(self._process_group, "shutdown", None)
        if callable(pg_shutdown):
            pg_shutdown()

    def __enter__(self) -> "MuLoCoController":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        self.close()
        return False

    def _create_torchft_manager(self, config: TorchFTReplicaConfig) -> tuple[Any, Any]:
        if _torchft is None:
            raise TorchFTUnavailableError(
                "TorchFT is required to construct MuLoCo communication; "
                "install torchft or inject a Manager-compatible object"
            )

        process_group = _torchft.ProcessGroupNCCL(timeout=config.timeout)
        manager = _torchft.Manager(
            pg=process_group,
            load_state_dict=self._load_controller_state_dict,
            state_dict=self._controller_state_dict,
            min_replica_size=config.min_replica_size,
            use_async_quorum=False,
            timeout=config.timeout,
            quorum_timeout=config.quorum_timeout,
            connect_timeout=config.connect_timeout,
            rank=config.rank,
            world_size=config.world_size,
            store_addr=config.store_addr,
            store_port=config.store_port,
            lighthouse_addr=config.lighthouse_addr,
            replica_id=config.replica_id,
            port=config.manager_port,
            hostname=config.hostname,
            init_sync=config.init_sync,
        )
        return process_group, manager

    def _controller_state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "local_step": self._local_step,
            "total_inner_steps": self._total_inner_steps,
            "outer_steps": self._outer_steps,
            "sync_every": self.sync_every,
            "outer_maximize": self.outer_maximize,
            "strict_commit": self._strict_commit,
            "shard_signature": self._shard_signature,
            "anchors": [anchor.detach().clone() for anchor in self._anchors],
            "outer_optimizer": self._outer_optimizer.state_dict(),
            "model_refresh_required": self._model_refresh_required,
        }

    @torch.no_grad()
    def _load_controller_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        version = int(state_dict.get("version", 0))
        if version != self._STATE_VERSION:
            raise ValueError(
                f"unsupported MuLoCo state version {version}; expected {self._STATE_VERSION}"
            )
        if int(state_dict["sync_every"]) != self.sync_every:
            raise ValueError(
                "MuLoCo sync_every changed on resume: "
                f"checkpoint={state_dict['sync_every']} current={self.sync_every}"
            )
        if bool(state_dict.get("outer_maximize", True)) != self.outer_maximize:
            raise ValueError(
                "MuLoCo outer_maximize changed on resume: "
                f"checkpoint={state_dict.get('outer_maximize', True)} "
                f"current={self.outer_maximize}"
            )
        if bool(state_dict.get("strict_commit", True)) != self._strict_commit:
            raise ValueError(
                "MuLoCo strict_commit changed on resume: "
                f"checkpoint={state_dict.get('strict_commit', True)} "
                f"current={self._strict_commit}"
            )
        loaded_signature = tuple(state_dict["shard_signature"])
        if loaded_signature != self._shard_signature:
            raise ValueError(
                "MuLoCo owned-shard layout differs from the checkpoint: "
                f"checkpoint={loaded_signature} current={self._shard_signature}"
            )

        loaded_anchors = state_dict["anchors"]
        if len(loaded_anchors) != len(self._anchors):
            raise ValueError(
                f"checkpoint has {len(loaded_anchors)} anchors, expected {len(self._anchors)}"
            )
        for index, (anchor, loaded) in enumerate(zip(self._anchors, loaded_anchors)):
            if tuple(loaded.shape) != tuple(anchor.shape):
                raise ValueError(
                    f"anchor {index} shape mismatch: checkpoint={tuple(loaded.shape)} "
                    f"current={tuple(anchor.shape)}"
                )
            anchor.copy_(loaded)

        self._outer_optimizer.load_state_dict(state_dict["outer_optimizer"])
        self._local_step = int(state_dict["local_step"])
        self._total_inner_steps = int(state_dict["total_inner_steps"])
        self._outer_steps = int(state_dict["outer_steps"])
        self._model_refresh_required = bool(
            state_dict.get("model_refresh_required", False)
        )
        self._signature_verified = False

    def _verify_corresponding_shards(self) -> None:
        assert self._manager is not None
        allgather = getattr(self._manager, "allgather", None)
        if not callable(allgather):
            logger.warning(
                "TorchFT Manager has no allgather(); skipping MuLoCo shard-signature "
                "verification"
            )
            self._signature_verified = True
            return

        signature = torch.tensor(
            self._shard_signature,
            dtype=torch.int64,
            device=self._params[0].device,
        )
        communication_probes.record_collective(
            "muloco_signature_allgather", signature.numel() * signature.element_size()
        )
        result = allgather(signature)
        gathered = result.wait() if hasattr(result, "wait") else result
        if isinstance(gathered, torch.Tensor):
            gathered_signatures = list(gathered.reshape(-1, signature.numel()))
        else:
            gathered_signatures = list(gathered)
        if not gathered_signatures:
            raise RuntimeError("MuLoCo shard-signature allgather returned no values")
        reference = gathered_signatures[0]
        if any(not torch.equal(reference, value) for value in gathered_signatures[1:]):
            raise RuntimeError(
                "MuLoCo corresponding ranks own different shard layouts: "
                f"local={self._shard_signature}, "
                f"gathered={[value.tolist() for value in gathered_signatures]}"
            )
        self._signature_verified = True

    @staticmethod
    def _compute_shard_signature(
        parameters: Sequence[torch.Tensor],
    ) -> tuple[int, int, int]:
        digest = hashlib.blake2b(digest_size=8)
        total_numel = 0
        for index, param in enumerate(parameters):
            total_numel += param.numel()
            metadata = (
                index,
                tuple(param.shape),
                tuple(param.stride()),
                str(param.dtype),
                param.numel(),
            )
            digest.update(repr(metadata).encode("ascii"))
        # Keep the digest in the positive signed-int64 range for NCCL all-gather.
        layout_hash = int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)
        return len(parameters), total_numel, layout_hash

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("MuLoCoController is closed")


__all__ = [
    "MuLoCoController",
    "TorchFTReplicaConfig",
    "TorchFTUnavailableError",
]
