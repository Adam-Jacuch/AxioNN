# src/axionn/nn/ssm.py

from typing import Optional
import jax.numpy as jnp
from axiom.core import Tensor, Axis, Bundle, TargetedBundle
from axiom.nn import ssm_op


def ssm_scan(
    x: Tensor,
    a: Tensor,
    seq_ax: Optional[Axis] = None,
    mode: str = "convex",
    decay_activation: str = "sigmoid",
    associative: bool = True,
) -> Tensor:
    """
    General, stable, and expressive state-space model (SSM) scan primitive.

    Computes a stable linear recurrence over the sequence axis using Axiom's scan engine.
    
    Formula:
        h_t = A_t * h_{t-1} + X'_t
    """
    # 1. Resolve sequence axis dynamically if not provided
    if seq_ax is None:
        seq_ax = next((ax for ax in x.topology if ax.name == "s"), None)
        if seq_ax is None:
            seq_ax = next((ax for ax in x.topology[:-1] if "batch" not in ax.name), x.topology[0])

    # 2. Determine decay gating A_t
    if mode == "linear":
        A_t = a
    else:
        if decay_activation == "sigmoid":
            A_t = a.sigmoid()
        elif decay_activation == "softplus":
            A_t = (-a.softplus()).exp()
        elif decay_activation == "none" or decay_activation is None:
            A_t = a
        else:
            raise ValueError(f"Unknown decay_activation: {decay_activation}")

    # 3. Determine input scaling X'_t
    if mode == "convex":
        X_gated = (1.0 - A_t) * x
    elif mode == "norm_preserving":
        X_gated = (1.0 - A_t**2).sqrt() * x
    elif mode in ("stable", "linear"):
        X_gated = x
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # 4. Perform the scan using TargetedBundle
    targeted_bundle = TargetedBundle(X_gated & A_t, (seq_ax,))

    if associative:
        return targeted_bundle.scan(ssm_op, associative=True)[0]
    
    # Sequential scan
    h_axes = tuple(ax for ax in X_gated.topology if ax != seq_ax)
    a_axes = tuple(ax for ax in A_t.topology if ax != seq_ax)

    h_init = Tensor(jnp.zeros(tuple(ax.size for ax in h_axes)), *h_axes)
    a_init = Tensor(jnp.ones(tuple(ax.size for ax in a_axes)), *a_axes)

    def sequential_step(carry, xt):
        X_new, A_new = ssm_op(carry, xt)
        return Bundle(X_new, A_new), Bundle(X_new, A_new)

    _, y_bundle = targeted_bundle.scan(sequential_step, init=Bundle(h_init, a_init), associative=False)
    return y_bundle.tensors[0]
