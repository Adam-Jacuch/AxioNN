# src/axionn/nn/attention.py

from typing import Optional
from axiom.core import Tensor, Axis, TargetedTensor
from axiom.nn import flash_attention

def GQA(
        x: Tensor,
        num_queries_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mask: Optional[Tensor] = None,
        tie_prefix: Optional[str] = None
) -> Tensor:
    """
    Grouped Query Attention module.
    Automatically orchestrates projection, head-splitting, key replication, and merging.
    """
    # Define internal runtime axes dynamically
    h_q = Axis("q_heads", num_queries_heads)
    h_kv = Axis("kv_heads", num_kv_heads)
    d_h = Axis("head_dim", head_dim)

    # Resolve the sequence axis dynamically from the input tensor topology
    seq_ax = None
    d_model = x.topology[-1]
    for ax in x.topology[:-1]:
        if "batch" in ax.name.lower():
            pass
        else:
            seq_ax = ax
    
    if seq_ax is None:
        seq_ax = x.topology[0]

    # 1. Project Q, K, V using Axiom bundles or targeted projections
    q_proj_axis = Axis("q_proj_dim", num_queries_heads * head_dim)
    k_proj_axis = Axis("k_proj_dim", num_kv_heads * head_dim)
    v_proj_axis = Axis("v_proj_dim", num_kv_heads * head_dim)

    q_proj = getattr(x, d_model.name).proj(q_proj_axis, tie=f"{tie_prefix}_q" if tie_prefix else None)
    k_proj = getattr(x, d_model.name).proj(k_proj_axis, tie=f"{tie_prefix}_k" if tie_prefix else None)
    v_proj = getattr(x, d_model.name).proj(v_proj_axis, tie=f"{tie_prefix}_v" if tie_prefix else None)

    # 2. Reshape into head topologies
    q_heads = getattr(q_proj, q_proj_axis.name).split(h_q, d_h)
    k_heads = getattr(k_proj, k_proj_axis.name).split(h_kv, d_h)
    v_heads = getattr(v_proj, v_proj_axis.name).split(h_kv, d_h)

    # 3. Invoke the ultra-optimized baseline dispatcher from your axiom.nn
    out = flash_attention(q_heads, k_heads, v_heads, seq_ax=seq_ax, head_ax=h_q, mask=mask)

    # 4. Collapse heads back down to original embedding shape
    out_proj_axis = Axis("out_proj_dim", num_queries_heads * head_dim)
    out_collapsed = TargetedTensor(out, (h_q, d_h)).merge(out_proj_axis)

    # 5. Output projection
    return getattr(out_collapsed, out_proj_axis.name).proj(d_model, tie=f"{tie_prefix}_out" if tie_prefix else None)


def MHA(
        x: Tensor,
        num_heads: int,
        head_dim: int,
        mask: Optional[Tensor] = None,
        tie_prefix: Optional[str] = None
) -> Tensor:
    """Multi-Head Attention realized as a special symmetric variant of GQA."""
    return GQA(
        x=x,
        num_queries_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        mask=mask,
        tie_prefix=tie_prefix
    )


def SWA(
        x: Tensor,
        num_heads: int,
        head_dim: int,
        window_size: int,
        causal: bool = True,
        tie_prefix: Optional[str] = None
) -> Tensor:
    """
    Sliding Window Attention (SWA).
    Efficiently restricts each token's attention to a local window.
    """
    import jax.numpy as jnp
    from axiom.core import Axis, Tensor
    
    # Resolve the sequence axis dynamically from the input tensor topology
    seq_ax = None
    for ax in x.topology[:-1]:
        if "batch" not in ax.name.lower():
            seq_ax = ax
            break
            
    if seq_ax is None:
        seq_ax = x.topology[0]
        
    N = seq_ax.size
    
    # Construct relative position mask
    q_pos = jnp.arange(N)[:, None]
    k_pos = jnp.arange(N)[None, :]
    
    if causal:
        mask_raw = (q_pos >= k_pos) & (q_pos - k_pos < window_size)
    else:
        mask_raw = jnp.abs(q_pos - k_pos) < window_size
        
    key_seq_ax = Axis(f"{seq_ax.name}_key", N)
    mask = Tensor(mask_raw, seq_ax, key_seq_ax)
    
    return GQA(
        x=x,
        num_queries_heads=num_heads,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        mask=mask,
        tie_prefix=tie_prefix
    )


def LinearAttention(
        x: Tensor,
        num_heads: int,
        head_dim: int,
        causal: bool = True,
        tie_prefix: Optional[str] = None
) -> Tensor:
    """
    Linear Attention (both Causal and Non-Causal).
    Replaces the softmax with a kernel feature map (elu(x) + 1) for O(N) complexity.
    """
    import jax
    import jax.numpy as jnp
    from axiom.core import Axis, Tensor, TargetedTensor

    # Define internal runtime axes dynamically
    h_q = Axis("q_heads", num_heads)
    h_kv = Axis("kv_heads", num_heads)
    d_h = Axis("head_dim", head_dim)

    # Resolve the sequence axis dynamically from the input tensor topology
    seq_ax = None
    d_model = x.topology[-1]
    for ax in x.topology[:-1]:
        if "batch" not in ax.name.lower():
            seq_ax = ax
            break
    
    if seq_ax is None:
        seq_ax = x.topology[0]

    # 1. Project Q, K, V
    q_proj_axis = Axis("q_proj_dim", num_heads * head_dim)
    k_proj_axis = Axis("k_proj_dim", num_heads * head_dim)
    v_proj_axis = Axis("v_proj_dim", num_heads * head_dim)

    q_proj = getattr(x, d_model.name).proj(q_proj_axis, tie=f"{tie_prefix}_q" if tie_prefix else None)
    k_proj = getattr(x, d_model.name).proj(k_proj_axis, tie=f"{tie_prefix}_k" if tie_prefix else None)
    v_proj = getattr(x, d_model.name).proj(v_proj_axis, tie=f"{tie_prefix}_v" if tie_prefix else None)

    # 2. Reshape into head topologies
    q_heads = getattr(q_proj, q_proj_axis.name).split(h_q, d_h)
    k_heads = getattr(k_proj, k_proj_axis.name).split(h_kv, d_h)
    v_heads = getattr(v_proj, v_proj_axis.name).split(h_kv, d_h)

    # 3. Apply ELU + 1 kernel feature map to Q and K
    q_raw = q_heads.unwrap()
    k_raw = k_heads.unwrap()
    v_raw = v_heads.unwrap()

    q_feat = jax.nn.elu(q_raw) + 1.0
    k_feat = jax.nn.elu(k_raw) + 1.0

    # 4. Compute Causal or Non-Causal Linear Attention
    if causal:
        # Causal: sum up to current timestep
        # Key-Value outer product: shape (batch..., seq, heads, d_h, d_h)
        kv = jnp.einsum("...shd,...shv->...shdv", k_feat, v_raw)
        
        # Cumulative sum along sequence dimension (axis -4 in kv)
        S = jnp.cumsum(kv, axis=-4)
        
        # Multiply Q by state S: shape (batch..., seq, heads, d_h)
        num = jnp.einsum("...shd,...shdv->...shv", q_feat, S)
        
        # Normalization denominator
        k_cumsum = jnp.cumsum(k_feat, axis=-3)  # axis -3 in k_feat is seq
        den = jnp.einsum("...shd,...shd->...sh", q_feat, k_cumsum)
    else:
        # Non-Causal: global sum
        # Key-Value state: shape (batch..., heads, d_h, d_h)
        S = jnp.einsum("...shd,...shv->...hdv", k_feat, v_raw)
        
        # Multiply Q by state S: shape (batch..., seq, heads, d_h)
        num = jnp.einsum("...shd,...hdv->...shv", q_feat, S)
        
        # Normalization denominator
        k_sum = jnp.sum(k_feat, axis=-3)  # axis -3 is seq
        den = jnp.einsum("...shd,...hd->...sh", q_feat, k_sum)

    # Divide by denominator with epsilon for numerical stability
    out_raw = num / (den[..., None] + 1e-6)

    # Wrap raw result back to Tensor
    out = Tensor(out_raw, *q_heads.topology)

    # 5. Collapse heads back down to original embedding shape
    out_proj_axis = Axis("out_proj_dim", num_heads * head_dim)
    out_collapsed = TargetedTensor(out, (h_q, d_h)).merge(out_proj_axis)

    # 6. Output projection
    return getattr(out_collapsed, out_proj_axis.name).proj(d_model, tie=f"{tie_prefix}_out" if tie_prefix else None)
