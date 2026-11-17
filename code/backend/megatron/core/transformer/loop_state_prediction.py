# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Decoupled recurrent-state and prediction readouts for loop transformers."""

from __future__ import annotations

import math
from typing import Callable, Optional

from torch import Tensor, nn
from torch.nn import functional as F


def rms_normalize(x: Tensor, eps: float) -> Tensor:
    """Parameter-free RMS normalization with FP32 reduction statistics."""

    inverse_rms = (x.float().square().mean(dim=-1, keepdim=True) + eps).rsqrt()
    return x * inverse_rms.to(dtype=x.dtype)


class DirectLinearStatePredictionSplit(nn.Module):
    """Project one loop-boundary state into independent state and prediction paths.

    Both paths are direct, full-rank, bias-free projections of the same normalized
    hidden state.  There is deliberately no residual bypass or activation.  Either
    projection can be skipped, which avoids dead work for zero-weight NTP exits and
    for the final loop visit, whose recurrent state is not consumed.
    """

    def __init__(
        self,
        hidden_size: int,
        init_method: Callable[[Tensor], object],
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")

        self.hidden_size = hidden_size
        self.eps = eps
        self.state_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prediction_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        # Use the model's ordinary hidden-to-hidden initialization twice.  The
        # distributions match, but sequential RNG draws make the two roles
        # independent from the first update onward.
        init_method(self.state_projection.weight)
        init_method(self.prediction_projection.weight)

    def forward(
        self,
        hidden_states: Tensor,
        *,
        project_state: bool = True,
        project_prediction: bool = True,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "state/prediction split input must have shape [sequence, batch, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if not project_state and not project_prediction:
            raise ValueError("at least one state/prediction projection must be requested.")

        normalized = rms_normalize(hidden_states, self.eps)
        state = self.state_projection(normalized) if project_state else None
        prediction = (
            self.prediction_projection(normalized) if project_prediction else None
        )
        return state, prediction

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, eps={self.eps}, bias=False"


class IdentityLinearStatePredictionSplit(nn.Module):
    """Identity-initialized linear readouts without boundary normalization.

    ``prediction_only=True`` leaves the recurrent carrier exactly unchanged and
    learns only ``p = W_p h``. Otherwise it learns both ``s = W_s h`` and
    ``p = W_p h``. Every learned map is full-rank, bias-free, and initialized
    to identity. The weights remain unconstrained and use the ordinary model
    learning rate; an attribute marks them for zero weight decay so AdamW does
    not move the identity origin toward the zero matrix.
    """

    def __init__(self, hidden_size: int, *, prediction_only: bool) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")

        self.hidden_size = hidden_size
        self.prediction_only = prediction_only
        self.state_projection = (
            None
            if prediction_only
            else nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.prediction_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        learned_projections = [self.prediction_projection]
        if self.state_projection is not None:
            learned_projections.insert(0, self.state_projection)
        for projection in learned_projections:
            nn.init.eye_(projection.weight)
            projection.weight.is_loop_state_prediction_identity_parameter = True

    def forward(
        self,
        hidden_states: Tensor,
        *,
        project_state: bool = True,
        project_prediction: bool = True,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "state/prediction split input must have shape [sequence, batch, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if not project_state and not project_prediction:
            raise ValueError("at least one state/prediction projection must be requested.")

        state = None
        if project_state:
            state = (
                hidden_states
                if self.state_projection is None
                else self.state_projection(hidden_states)
            )
        prediction = (
            self.prediction_projection(hidden_states) if project_prediction else None
        )
        return state, prediction

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, prediction_only={self.prediction_only}, "
            "normalization=none, initialization=identity, bias=False"
        )


class PerVisitIdentityLinearStatePredictionSplit(nn.Module):
    """Use visit-specific identity-initialized prediction readouts.

    Visit ``v`` owns its prediction map ``W_p[v]``.  The recurrent maps that feed
    visit ``v + 1`` can either be independent ``W_s[v]`` maps or one shared
    ``W_s`` map.  All maps are full-rank and bias-free, and they are initialized
    independently to the same identity function.  Consequently either mode is
    functionally identical to the shared dual-identity split at initialization.

    ``loop_visit`` is one-based to match the loop execution metadata used by
    :class:`TransformerBlock`.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_visits: int,
        share_state_projection: bool = False,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}.")
        if num_visits < 2:
            raise ValueError(f"num_visits must be at least 2, got {num_visits}.")

        self.hidden_size = hidden_size
        self.num_visits = num_visits
        self.share_state_projection = share_state_projection
        num_state_projections = 1 if share_state_projection else num_visits - 1
        self.state_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_state_projections)
        )
        self.prediction_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_visits)
        )

        for projection in (*self.state_projections, *self.prediction_projections):
            nn.init.eye_(projection.weight)
            projection.weight.is_loop_state_prediction_identity_parameter = True

    def forward(
        self,
        hidden_states: Tensor,
        *,
        loop_visit: int,
        project_state: bool = True,
        project_prediction: bool = True,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "state/prediction split input must have shape [sequence, batch, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if not project_state and not project_prediction:
            raise ValueError("at least one state/prediction projection must be requested.")
        if loop_visit < 1 or loop_visit > self.num_visits:
            raise ValueError(
                f"loop_visit must lie in [1, {self.num_visits}], got {loop_visit}."
            )
        if project_state and loop_visit == self.num_visits:
            raise ValueError("the final loop visit has no recurrent-state consumer.")

        projection_index = loop_visit - 1
        state_projection_index = 0 if self.share_state_projection else projection_index
        state = (
            self.state_projections[state_projection_index](hidden_states)
            if project_state
            else None
        )
        prediction = (
            self.prediction_projections[projection_index](hidden_states)
            if project_prediction
            else None
        )
        return state, prediction

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_visits={self.num_visits}, "
            "normalization=none, initialization=identity, bias=False, "
            f"state_shared_across_visits={self.share_state_projection}, "
            "prediction_shared_across_visits=False"
        )


class _ProjectedResidualMLP(nn.Module):
    """A bias-free residual MLP whose skip lives in projected branch space."""

    def __init__(
        self,
        hidden_size: int,
        bottleneck_size: int,
        init_method: Callable[[Tensor], object],
        output_layer_init_method: Callable[[Tensor], object],
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(
            hidden_size, bottleneck_size, bias=False
        )
        self.output_projection = nn.Linear(
            bottleneck_size, hidden_size, bias=False
        )
        init_method(self.input_projection.weight)
        output_layer_init_method(self.output_projection.weight)

    def forward(self, projected: Tensor) -> Tensor:
        return projected + self.output_projection(
            F.silu(self.input_projection(projected))
        )


class ProjectedResidualMLPStatePredictionSplit(nn.Module):
    """Identity-project, then refine each state/prediction branch with an MLP.

    The recurrent path owns one projection and one residual MLP shared by all
    non-final visits.  Every visit owns an independent prediction projection
    and residual MLP.  The mandatory full-rank projections are initialized to
    identity and receive no boundary normalization.  Importantly, the residual
    skip is the *projected* branch value rather than the raw hidden state::

        q_s = Q_s h;       s = q_s + F_s(q_s)
        q_p[v] = Q_p[v] h; p[v] = q_p[v] + F_p[v](q_p[v])

    ``F_s`` is shared across visits, while ``F_p[v]`` is visit specific.  Each
    bias-free SiLU MLP uses a d/4 bottleneck and the Transformer's ordinary
    input/output initialization methods.  Selective routing avoids both the
    unused final recurrent branch and prediction branches for zero-weight exits.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        num_visits: int,
        init_method: Callable[[Tensor], object],
        output_layer_init_method: Callable[[Tensor], object],
    ) -> None:
        super().__init__()
        if hidden_size < 4 or hidden_size % 4 != 0:
            raise ValueError(
                "hidden_size must be a positive multiple of four for the d/4 "
                f"residual-MLP bottleneck, got {hidden_size}."
            )
        if num_visits < 2:
            raise ValueError(f"num_visits must be at least 2, got {num_visits}.")

        self.hidden_size = hidden_size
        self.num_visits = num_visits
        self.bottleneck_size = hidden_size // 4
        self.state_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prediction_projections = nn.ModuleList(
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(num_visits)
        )
        self.state_residual_mlp = _ProjectedResidualMLP(
            hidden_size,
            self.bottleneck_size,
            init_method,
            output_layer_init_method,
        )
        self.prediction_residual_mlps = nn.ModuleList(
            _ProjectedResidualMLP(
                hidden_size,
                self.bottleneck_size,
                init_method,
                output_layer_init_method,
            )
            for _ in range(num_visits)
        )

        for projection in (self.state_projection, *self.prediction_projections):
            nn.init.eye_(projection.weight)
            projection.weight.is_loop_state_prediction_identity_parameter = True

    def forward(
        self,
        hidden_states: Tensor,
        *,
        loop_visit: int,
        project_state: bool = True,
        project_prediction: bool = True,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "state/prediction split input must have shape [sequence, batch, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if not project_state and not project_prediction:
            raise ValueError("at least one state/prediction projection must be requested.")
        if loop_visit < 1 or loop_visit > self.num_visits:
            raise ValueError(
                f"loop_visit must lie in [1, {self.num_visits}], got {loop_visit}."
            )
        if project_state and loop_visit == self.num_visits:
            raise ValueError("the final loop visit has no recurrent-state consumer.")

        state = None
        if project_state:
            projected_state = self.state_projection(hidden_states)
            state = self.state_residual_mlp(projected_state)

        prediction = None
        if project_prediction:
            prediction_index = loop_visit - 1
            projected_prediction = self.prediction_projections[prediction_index](
                hidden_states
            )
            prediction = self.prediction_residual_mlps[prediction_index](
                projected_prediction
            )
        return state, prediction

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, bottleneck_size={self.bottleneck_size}, "
            f"num_visits={self.num_visits}, normalization=none, bias=False, "
            "projection_initialization=identity, projected_residual=True, "
            "state_shared_across_visits=True, prediction_shared_across_visits=False"
        )


class DirectMLPStatePredictionSplit(nn.Module):
    """Nonlinearly decouple recurrent-state and prediction paths.

    Each path is an independent, bias-free ``d -> d/2 -> d`` SiLU MLP.  The
    bottleneck makes the two MLPs together exactly parameter- and matmul-FLOP
    matched to the two ``d -> d`` projections in
    :class:`DirectLinearStatePredictionSplit`.  There is no residual bypass.
    Xavier-normal initialization is deliberately non-zero because, without a
    bypass, zero-initializing either output projection would erase that path.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if hidden_size < 2 or hidden_size % 2 != 0:
            raise ValueError(
                "hidden_size must be a positive even integer for an equal-parameter "
                f"d/2 bottleneck, got {hidden_size}."
            )
        if not math.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be finite and positive, got {eps}.")

        self.hidden_size = hidden_size
        self.bottleneck_size = hidden_size // 2
        self.eps = eps
        self.state_input_projection = nn.Linear(
            hidden_size, self.bottleneck_size, bias=False
        )
        self.state_output_projection = nn.Linear(
            self.bottleneck_size, hidden_size, bias=False
        )
        self.prediction_input_projection = nn.Linear(
            hidden_size, self.bottleneck_size, bias=False
        )
        self.prediction_output_projection = nn.Linear(
            self.bottleneck_size, hidden_size, bias=False
        )

        for projection in (
            self.state_input_projection,
            self.state_output_projection,
            self.prediction_input_projection,
            self.prediction_output_projection,
        ):
            nn.init.xavier_normal_(projection.weight, gain=1.0)

    @staticmethod
    def _project(
        normalized: Tensor,
        input_projection: nn.Linear,
        output_projection: nn.Linear,
    ) -> Tensor:
        return output_projection(F.silu(input_projection(normalized)))

    def forward(
        self,
        hidden_states: Tensor,
        *,
        project_state: bool = True,
        project_prediction: bool = True,
    ) -> tuple[Optional[Tensor], Optional[Tensor]]:
        if hidden_states.ndim != 3:
            raise ValueError(
                "state/prediction split input must have shape [sequence, batch, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError(
                f"expected hidden size {self.hidden_size}, got {hidden_states.shape[-1]}."
            )
        if not project_state and not project_prediction:
            raise ValueError("at least one state/prediction projection must be requested.")

        normalized = rms_normalize(hidden_states, self.eps)
        state = (
            self._project(
                normalized,
                self.state_input_projection,
                self.state_output_projection,
            )
            if project_state
            else None
        )
        prediction = (
            self._project(
                normalized,
                self.prediction_input_projection,
                self.prediction_output_projection,
            )
            if project_prediction
            else None
        )
        return state, prediction

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, bottleneck_size={self.bottleneck_size}, "
            f"eps={self.eps}, activation=silu, bias=False"
        )
