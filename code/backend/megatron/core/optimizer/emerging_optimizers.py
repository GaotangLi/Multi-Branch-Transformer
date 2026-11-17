# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Emerging optimizer registry.

To add a new emerging optimizer:
  1. Define its optimizer class (or import it).
  2. Write its ``_<name>_init_state_fn`` and ``_<name>_config_to_kwargs``.
  3. Add an ``EmergingOptimizerEntry`` to ``_EMERGING_OPTIMIZERS`` at the bottom.
"""

import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, get_args

import torch
from torch.optim.optimizer import ParamsT

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.utils import get_pg_size, log_single_rank

from .gpa import FixedGPAParamSequence, GPAAdamW
from .marin_hyperball import (
    MARIN_HYPERBALL_ROLE,
    MARIN_SCALAR_LR_RATIO,
    MarinHyperball,
    ROLE_ADAMH,
    classify_marin_hyperball_parameter,
)
from .optimizer_config import ParamKey, ParamPredicate, ParamWithNamePredicate

try:
    from emerging_optimizers import registry
    from emerging_optimizers.orthogonalized_optimizers import (
        AdaptiveMuon,
        OrthogonalizedOptimizer,
        get_muon_scale_factor,
    )
    from emerging_optimizers.orthogonalized_optimizers.muon_utils import NSCoeffT, newton_schulz_tp

    # It is necessary to import optimizers for the registry to work.
    from emerging_optimizers.scalar_optimizers import Lion  # pylint: disable=unused-import
    from emerging_optimizers.soap import SOAP  # pylint: disable=unused-import

    HAVE_EMERGING_OPTIMIZERS = True
except ImportError:
    HAVE_EMERGING_OPTIMIZERS = False
    OrthogonalizedOptimizer = object
    AdaptiveMuon = object


logger = logging.getLogger(__name__)


def get_supported_coefficient_types() -> tuple[str, ...]:
    """Return the coefficient types supported by the installed emerging_optimizers.

    Reads the members of the ``NSCoeffT`` Literal type so that new types
    added upstream are automatically available without code changes here.
    """
    assert (
        HAVE_EMERGING_OPTIMIZERS
    ), "emerging_optimizers >= 0.2 is required for NSCoeffT. Please install or upgrade it."
    return get_args(NSCoeffT)


def validate_coefficient_type(coefficient_type: str) -> None:
    """Raise ``ValueError`` if *coefficient_type* is not supported."""
    supported = get_supported_coefficient_types()
    if coefficient_type not in supported:
        raise ValueError(
            f"Unsupported muon coefficient type '{coefficient_type}'. "
            f"Supported types: {supported}"
        )


# ===========================================================================
# Registry dataclass and public API
# ===========================================================================


def _eopt_init_state_fn(opt, config=None):
    """Initialize emerging optimizer state for torch_dist checkpoint format."""
    for group in opt.param_groups:
        # Checkpoint init needs state for all parameters, including those without grads yet.
        opt._init_group(group, skip_non_grad_params=False)


def _default_param_overrides_factory() -> Dict[ParamKey, Dict[str, Any]]:
    """Default param overrides: route non-linear/embedding params to Adam."""
    return {
        ParamKey(
            predicate=ParamPredicate(name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding)
        ): {'optimizer': 'adam'}
    }


@dataclass
class EmergingOptimizerEntry:
    """Everything needed to create and configure an emerging optimizer.

    Attributes:
        optimizer_cls: The torch optimizer class.
        init_state_fn: Lazily initialises optimizer state (needed for checkpoint formats).
        config_to_kwargs: ``(config, model_chunks, pg_collection) -> dict`` of constructor kwargs.
        default_param_overrides: Per-parameter config overrides applied automatically
            (e.g. route non-linear params to Adam).
        requires_external_package: Whether construction requires the optional
            ``emerging_optimizers`` dependency.
    """

    optimizer_cls: type
    init_state_fn: Callable = _eopt_init_state_fn
    config_to_kwargs: Callable | None = None
    default_param_overrides: Dict[ParamKey, Dict[str, Any]] = field(
        default_factory=_default_param_overrides_factory
    )
    requires_external_package: bool = True


def _create_emerging_optimizer(config, param_groups, eopt_name, model_chunks, pg_collection):
    """Instantiate an emerging optimizer and return it with its init_state_fn."""
    entry = _EMERGING_OPTIMIZERS[eopt_name]
    if entry.config_to_kwargs is not None:
        eopt_kwargs = entry.config_to_kwargs(config, model_chunks, pg_collection)
    else:
        eopt_kwargs = _default_adam_based_eopt_config_to_kwargs(
            eopt_name, config, model_chunks, pg_collection
        )
    optimizer = entry.optimizer_cls(param_groups, **eopt_kwargs)
    return optimizer, entry.init_state_fn


# ===========================================================================
# Shared helpers
# ===========================================================================


def _is_nonlinear_or_embedding(param):
    """True for parameters that should NOT use the emerging optimizer."""
    return getattr(param, 'is_embedding_or_output_parameter', False) or len(param.shape) != 2


def _get_qkv_split_shapes(model_cfg) -> List[int]:
    """Compute fused attention projection split widths from model config."""
    query_width = model_cfg.num_attention_heads // model_cfg.num_query_groups * model_cfg.kv_channels
    split_shapes = [
        query_width,
        model_cfg.kv_channels,
        model_cfg.kv_channels,
    ]
    if model_cfg.attention_output_gate:
        # Gated attention stores fused projections as query, gate, key, value.
        split_shapes.insert(1, query_width)
    return split_shapes


# ===========================================================================
# Registry – populated below only when emerging_optimizers is installed.
# ===========================================================================

_EMERGING_OPTIMIZERS: Dict[str, EmergingOptimizerEntry] = {}


# ===========================================================================
# Muon
# ===========================================================================


class TensorParallelMuon(OrthogonalizedOptimizer):
    """Tensor Parallel Muon optimizer."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | tuple[int, ...] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
    ) -> None:
        if num_ns_steps < 1:
            raise ValueError(f"num_ns_steps must be at least 1, got {num_ns_steps}")

        def scaled_orthogonalize_fn(
            grad: torch.Tensor,
            tp_group: torch.distributed.ProcessGroup,
            partition_dim: int | None = None,
        ) -> torch.Tensor:
            log_single_rank(
                logger,
                logging.DEBUG,
                f'Orthogonalizing grad with {num_ns_steps} steps, '
                f'{coefficient_type} coefficient, '
                f'{scale_mode} scale mode, extra_scale_factor={extra_scale_factor}',
            )
            size = [grad.size(-2), grad.size(-1)]
            if partition_dim is not None:
                size[partition_dim] *= get_pg_size(tp_group)
            orth_grad = newton_schulz_tp(
                grad,
                steps=num_ns_steps,
                coefficient_type=coefficient_type,
                tp_group=tp_group,
                partition_dim=partition_dim,
                tp_mode="duplicated" if tp_mode == "blockwise" else tp_mode,
            )
            scale_factor = get_muon_scale_factor(size[0], size[1], mode=scale_mode)
            return orth_grad * scale_factor * extra_scale_factor

        self.pg_collection = pg_collection
        self.tp_mode = tp_mode
        self.split_qkv = split_qkv
        self.is_qkv_fn = is_qkv_fn
        self.qkv_split_shapes = qkv_split_shapes

        weight_decay_method = "decoupled" if use_decoupled_weight_decay else "l2"
        # Use explicit class call instead of super() so that subclasses with
        # multiple inheritance (e.g. TensorParallelAdaptiveMuon) don't route
        # through an intermediate class that doesn't accept scaled_orthogonalize_fn.
        OrthogonalizedOptimizer.__init__(
            self,
            params,
            lr,
            momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            weight_decay_method=weight_decay_method,
            fp32_matmul_prec=fp32_matmul_prec,
            scaled_orthogonalize_fn=scaled_orthogonalize_fn,
        )

    def orthogonalize(self, p: torch.Tensor, grad: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Orthogonalize the momentum.

        Args:
            p: The parameter tensor. i is necessary to pass param tensor in addition to
                momentum because a lot of information is only available in the param tensor,
                attributes for example.
            grad: The momentum tensor.

        Returns:
            The orthogonalized gradient tensor.
        """
        # TODO(deyuf): switch to group
        if self.pg_collection:
            tp_group = (
                self.pg_collection.expt_tp
                if getattr(p, 'expert_tp', False)
                else self.pg_collection.tp
            )
        else:
            tp_group = None
        partition_dim = None if self.tp_mode == "blockwise" else getattr(p, "partition_dim", None)
        if partition_dim == -1:
            partition_dim = None

        if self.split_qkv and self.is_qkv_fn(p):  # type: ignore[misc]
            grad_shape = grad.shape
            log_single_rank(
                logger,
                logging.DEBUG,
                f'qkv split grad shape {grad_shape}, ' f'split shapes {self.qkv_split_shapes}',
            )
            num_query_groups = grad_shape[0] // sum(self.qkv_split_shapes)
            qkv_grads = torch.split(
                grad.view(num_query_groups, sum(self.qkv_split_shapes), -1),
                self.qkv_split_shapes,
                dim=1,
            )
            qkv_grads = [g.reshape(-1, grad_shape[-1]) for g in qkv_grads]

            qkv_grads = [
                self.scaled_orthogonalize_fn(g, tp_group, partition_dim).view(
                    num_query_groups, -1, grad_shape[-1]
                )
                for g in qkv_grads
            ]
            grad = torch.cat(qkv_grads, dim=1).view(grad_shape)
        else:
            grad = self.scaled_orthogonalize_fn(grad, tp_group, partition_dim)
        return grad


class GPATensorParallelMuon(FixedGPAParamSequence, TensorParallelMuon):
    """Tensor-parallel Muon used as the GPA base update on the ``z`` sequence."""

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | tuple[int, ...] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        train_interp_coeff: float = 0.7,
        eval_interp_coeff: float = 0.9967,
    ) -> None:
        FixedGPAParamSequence.validate_interp_coefficients(
            train_interp_coeff, eval_interp_coeff
        )
        if not use_decoupled_weight_decay:
            raise ValueError("GPA-Muon requires decoupled weight decay")
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        for group in self.param_groups:
            for param in group["params"]:
                if param.ndim != 2:
                    raise ValueError(
                        "GPATensorParallelMuon only accepts matrix parameters; "
                        f"got shape {tuple(param.shape)}"
                    )
            group.setdefault("optimizer", "gpa_muon")
            group.setdefault("train_interp_coeff", train_interp_coeff)
            group.setdefault("eval_interp_coeff", eval_interp_coeff)
            group.setdefault("train_mode", False)

    @torch.no_grad()
    def initialize_state(self) -> None:
        """Materialize Muon momentum and the GPA ``z`` buffer for every parameter."""
        for group in self.param_groups:
            self._init_group(group, skip_non_grad_params=False)
        self._initialize_gpa_buffers()

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[Any]:
        """Apply Muon to ``z``, then update the GPA training iterate ``y``."""
        self.initialize_state()
        self._require_gpa_train_mode()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is not None and param.grad.is_sparse:
                    raise RuntimeError("GPATensorParallelMuon does not support sparse gradients")

        active_params = []
        for group in self.param_groups:
            mu_y = group["train_interp_coeff"]
            mu_x = group["eval_interp_coeff"]
            z_new_coeff = 1.0 - mu_x * mu_y
            decay = 1.0 - group["lr"] * group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                z = self.state[param]["z"]
                y = param.data
                # Stage the old-z terms (after decoupled decay, as in GPA-AdamW)
                # so Muon can update z without another model-sized temporary.
                y.mul_(mu_x).add_(z, alpha=-mu_x * (1.0 - mu_y) * decay)
                active_params.append((param, y, z_new_coeff))
                param.data = z

        succeeded = False
        try:
            TensorParallelMuon.step(self)
            succeeded = True
        finally:
            for param, y, z_new_coeff in active_params:
                updated_z = param.data
                self.state[param]["z"] = updated_z
                param.data = y
                if succeeded:
                    param.add_(updated_z, alpha=z_new_coeff)

        return loss


class TensorParallelAdaptiveMuon(TensorParallelMuon, AdaptiveMuon):
    """Tensor Parallel Adaptive Muon optimizer.

    This class extends Muon by adding AdamW-style or NorMuon-style second moment
    accumulation after orthogonalization. This idea was first explored in D.E. Carlson,
    E. Collins, Ya-Ping Hsieh, L. Carin, and V. Cevher. *Preconditioned spectral
    descent for deep learning.* In Advances in neural information processing systems 28 (2015).
    The step() method is overridden to include second moment normalization logic.

    Args:
        params: Iterable of parameters to optimize or dicts defining parameter groups.
        lr: Learning rate.
        momentum: The exponential decay rate for momentum.
        nesterov: Whether to use Nesterov momentum.
        weight_decay: Weight decay coefficient.
        use_decoupled_weight_decay: Whether to use decoupled weight decay.
        split_qkv: Whether to split QKV weights for orthogonalization.
        is_qkv_fn: Function to determine if a tensor is a QKV weight.
        qkv_split_shapes: Shapes for splitting QKV weights.
        fp32_matmul_prec: Precision for FP32 matrix multiplication.
        coefficient_type: The type of coefficient set to use for the Newton-Schulz iteration.
        num_ns_steps: The number of iteration steps to use in the Newton-Schulz iteration.
        scale_mode: The type of scale factor to use for the update.
        extra_scale_factor: The additional scale factor to use for the update.
        pg_collection: Process group collection for distributed training.
        tp_mode: Tensor parallel mode ("blockwise", "duplicated", or "distributed").
        moment2_method: Method for second moment accumulation ("adamuon" or "normuon").
        beta2: The exponential decay rate for second moment.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        weight_decay: float = 0.01,
        use_decoupled_weight_decay: bool = True,
        split_qkv: bool = False,
        is_qkv_fn: Callable[[torch.Tensor], bool] | None = None,
        qkv_split_shapes: list[int] | tuple[int, ...] | None = None,
        fp32_matmul_prec: str = "medium",
        coefficient_type: str = "quintic",
        num_ns_steps: int = 5,
        scale_mode: str = "spectral",
        extra_scale_factor: float = 1.0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        tp_mode: Literal["blockwise", "duplicated", "distributed"] = "duplicated",
        moment2_method: Literal["adamuon", "normuon"] = "adamuon",
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        TensorParallelMuon.__init__(
            self,
            params,
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            weight_decay=weight_decay,
            use_decoupled_weight_decay=use_decoupled_weight_decay,
            split_qkv=split_qkv,
            is_qkv_fn=is_qkv_fn,
            qkv_split_shapes=qkv_split_shapes,
            fp32_matmul_prec=fp32_matmul_prec,
            coefficient_type=coefficient_type,
            num_ns_steps=num_ns_steps,
            scale_mode=scale_mode,
            extra_scale_factor=extra_scale_factor,
            pg_collection=pg_collection,
            tp_mode=tp_mode,
        )
        self.moment2_method = moment2_method

        for group in self.param_groups:
            group.setdefault("beta2", beta2)
            group.setdefault("eps", eps)

    @torch.no_grad()  # type: ignore[misc]
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Step function"""
        return AdaptiveMuon.step(self, closure)


def _kwargs_from_config(optimizer_cls: type, prefix: str, config) -> Dict[str, Any]:
    """Match ``optimizer_cls.__init__`` parameters to config attributes.

    For each init parameter, looks for ``{prefix}_{name}`` on *config* first,
    then falls back to ``{name}`` (unprefixed).  ``self`` and ``params`` are
    always skipped.
    """
    skip_params = {"self", "params"}
    sig = inspect.signature(optimizer_cls.__init__)
    kwargs: Dict[str, Any] = {}
    for name in sig.parameters:
        if name in skip_params:
            continue
        prefixed = f"{prefix}_{name}"
        if hasattr(config, prefixed):
            kwargs[name] = getattr(config, prefixed)
        elif hasattr(config, name):
            kwargs[name] = getattr(config, name)
    return kwargs


def _muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelMuon constructor kwargs."""
    kwargs = _kwargs_from_config(TensorParallelMuon, "muon", config)
    kwargs["is_qkv_fn"] = lambda p: getattr(p, "is_qkv", False)
    kwargs["qkv_split_shapes"] = _get_qkv_split_shapes(model_chunks[0].config)
    kwargs["pg_collection"] = pg_collection
    return kwargs


def _adaptive_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert OptimizerConfig to TensorParallelAdaptiveMuon constructor kwargs."""
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(_kwargs_from_config(TensorParallelAdaptiveMuon, "adaptive_muon", config))
    return kwargs


def _gpa_adamw_init_state_fn(opt, config=None):
    """Materialize all model-shaped GPA state before distributed checkpoint loading."""
    opt.initialize_state()


def _gpa_adamw_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert the shared Adam and GPA settings to GPAAdamW constructor arguments."""
    if not config.decoupled_weight_decay:
        raise ValueError("GPA-AdamW requires decoupled weight decay")
    return {
        "lr": config.lr,
        "eps": config.adam_eps,
        "beta1": config.adam_beta1,
        "beta2": config.adam_beta2,
        "weight_decay": config.weight_decay,
        "train_interp_coeff": config.gpa_train_interp_coeff,
        "eval_interp_coeff": config.gpa_eval_interp_coeff,
    }


def _gpa_muon_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert shared Muon and GPA settings to GPA-Muon constructor arguments."""
    if not config.decoupled_weight_decay:
        raise ValueError("GPA-Muon requires decoupled weight decay")
    kwargs = _muon_config_to_kwargs(config, model_chunks, pg_collection)
    kwargs.update(
        {
            "train_interp_coeff": config.gpa_train_interp_coeff,
            "eval_interp_coeff": config.gpa_eval_interp_coeff,
        }
    )
    return kwargs


def _marin_hyperball_config_to_kwargs(config, model_chunks, pg_collection) -> Dict[str, Any]:
    """Convert the exact Marin first-stage settings to optimizer arguments."""

    del pg_collection  # TP=1 is validated by the factory; no collectives are used in step().
    return {
        "lr": config.lr,
        "betas": (config.adam_beta1, config.adam_beta2),
        "eps": config.adam_eps,
        "momentum": config.marin_hyperball_momentum,
        "nesterov": config.marin_hyperball_nesterov,
        "num_ns_steps": config.marin_hyperball_num_ns_steps,
        "muon_eps": config.marin_hyperball_muon_eps,
        "qkv_split_shapes": _get_qkv_split_shapes(model_chunks[0].config),
        "scalar_lr_ratio": MARIN_SCALAR_LR_RATIO,
    }


def get_marin_hyperball_role_overrides(
    config, model_chunks
) -> Dict[ParamKey, Dict[str, Any]]:
    """Build complete, role-aware parameter overrides for Marin Hyperball."""

    model_config = model_chunks[0].config
    role_counts: Dict[str, int] = {}
    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad:
                continue
            role = classify_marin_hyperball_parameter(name, param, model_config)
            role_counts[role] = role_counts.get(role, 0) + 1

    if not role_counts:
        raise ValueError("Marin Hyperball found no trainable parameters on this rank")
    if (
        getattr(model_config, "pipeline_model_parallel_size", 1) == 1
        and ROLE_ADAMH not in role_counts
    ):
        raise ValueError(
            "Marin Hyperball found no untied output_layer.weight for the required AdamH role"
        )
    log_single_rank(
        logger,
        # The default Megatron logging configuration suppresses this module's
        # INFO records.  Keep the exhaustive role assignment visible in every
        # production log so a run cannot silently use the wrong transform.
        logging.WARNING,
        f"Marin Hyperball parameter role counts on this rank: {role_counts}",
    )

    overrides: Dict[ParamKey, Dict[str, Any]] = {}
    for role in sorted(role_counts):
        predicate = ParamWithNamePredicate(
            name=f"marin_hyperball_{role}",
            fn=lambda param, name, expected=role: (
                classify_marin_hyperball_parameter(name, param, model_config) == expected
            ),
        )
        override: Dict[str, Any] = {MARIN_HYPERBALL_ROLE: role}
        if role == "adam":
            override["max_lr"] = config.lr * MARIN_SCALAR_LR_RATIO
            if config.min_lr is not None:
                override["min_lr"] = config.min_lr * MARIN_SCALAR_LR_RATIO
        overrides[ParamKey(with_name_predicate=predicate)] = override
    return overrides


def _default_adam_based_eopt_config_to_kwargs(
    eopt_name, config, model_chunks, pg_collection
) -> Dict[str, Any]:
    """Convert OptimizerConfig to default emerging optimizer constructor kwargs."""
    kwargs = _kwargs_from_config(registry.get_optimizer_cls(eopt_name), eopt_name, config)
    kwargs["betas"] = (config.adam_beta1, config.adam_beta2)
    return kwargs


# -----------------------------------------------------------------------
# Register emerging optimizers
# -----------------------------------------------------------------------
_EMERGING_OPTIMIZERS.update(
    {
        'gpa_adamw': EmergingOptimizerEntry(
            optimizer_cls=GPAAdamW,
            init_state_fn=_gpa_adamw_init_state_fn,
            config_to_kwargs=_gpa_adamw_config_to_kwargs,
            default_param_overrides={},
            requires_external_package=False,
        ),
        'gpa_muon': EmergingOptimizerEntry(
            optimizer_cls=GPATensorParallelMuon,
            init_state_fn=_gpa_adamw_init_state_fn,
            config_to_kwargs=_gpa_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'gpa_adamw'}
            },
        ),
        'muon': EmergingOptimizerEntry(
            optimizer_cls=TensorParallelMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
        "marin_hyperball": EmergingOptimizerEntry(
            optimizer_cls=MarinHyperball,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_marin_hyperball_config_to_kwargs,
            default_param_overrides={},
            requires_external_package=False,
        ),
        "adaptive_muon": EmergingOptimizerEntry(
            optimizer_cls=TensorParallelAdaptiveMuon,
            init_state_fn=_eopt_init_state_fn,
            config_to_kwargs=_adaptive_muon_config_to_kwargs,
            default_param_overrides={
                ParamKey(
                    predicate=ParamPredicate(
                        name="nonlinear_or_embedding", fn=_is_nonlinear_or_embedding
                    )
                ): {'optimizer': 'adam'}
            },
        ),
    }
)

# Register soap with default config
# TODO(skyw): register all emerging optimizers.
if HAVE_EMERGING_OPTIMIZERS:
    for eopt_name in registry.get_optimizer_name_list():
        if eopt_name in _EMERGING_OPTIMIZERS:
            # skip already registered local versions, e.g. TensorParallel versions.
            continue
        _EMERGING_OPTIMIZERS[eopt_name] = EmergingOptimizerEntry(
            optimizer_cls=registry.get_optimizer_cls(eopt_name)
        )
