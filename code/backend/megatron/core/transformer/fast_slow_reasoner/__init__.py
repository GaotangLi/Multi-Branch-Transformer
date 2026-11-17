"""Opt-in Fast--Slow Reasoner building blocks.

Importing this package has no side effects. ``TransformerBlock`` constructs it
only when the explicit ``fast_slow_reasoner`` configuration flag is enabled.
"""

from .module import FastSlowReasoner, reduce_noninterleaved_rope_by_two

__all__ = ["FastSlowReasoner", "reduce_noninterleaved_rope_by_two"]
