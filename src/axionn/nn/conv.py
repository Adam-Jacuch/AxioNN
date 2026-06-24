# src/axionn/nn/conv.py

from typing import Union, Tuple, Optional
from axiom.core import Tensor, Axis, TargetedTensor


def conv(
    x: Tensor,
    in_channel_axes: Union[Axis, Tuple[Axis, ...]],
    out_channel_axes: Union[Axis, Tuple[Axis, ...]],
    spatial_axes: Union[Axis, Tuple[Axis, ...]],
    kernel_sizes: Union[int, Tuple[int, ...]],
    strides: Optional[Union[int, Tuple[int, ...]]] = None,
    padding: str = "same",
    causal: bool = False,
    bias: bool = True,
    tie_prefix: Optional[str] = None,
) -> Tensor:
    """
    Unified n-Dimensional Convolution primitive.

    Supports convolving over arbitrary numbers of spatial dimensions, merging arbitrary 
    input channel topologies, and projecting into arbitrary output channel topologies.
    """
    # 1. Normalize inputs to tuples
    in_channel_axes = (in_channel_axes,) if isinstance(in_channel_axes, Axis) else tuple(in_channel_axes)
    out_channel_axes = (out_channel_axes,) if isinstance(out_channel_axes, Axis) else tuple(out_channel_axes)
    spatial_axes = (spatial_axes,) if isinstance(spatial_axes, Axis) else tuple(spatial_axes)
    
    n_dim = len(spatial_axes)
    kernel_sizes = (kernel_sizes,) * n_dim if isinstance(kernel_sizes, int) else tuple(kernel_sizes)
    strides = (1,) * n_dim if strides is None else ((strides,) * n_dim if isinstance(strides, int) else tuple(strides))

    # 2. Pad & Unfold for each spatial dimension using TargetedTensor
    kernel_axes = []
    for ax, k, s in zip(spatial_axes, kernel_sizes, strides):
        if padding == "same":
            before, after = (k - 1, 0) if causal else ((k - 1) // 2, (k - 1) - (k - 1) // 2)
        elif padding == "valid":
            before, after = 0, 0
        else:
            raise ValueError(f"Unknown padding: {padding}")
            
        curr_ax = next(a for a in x.topology if a.name == ax.name)
        if before > 0 or after > 0:
            x = TargetedTensor(x, (curr_ax,)).pad((before, after))
            curr_ax = next(a for a in x.topology if a.name == ax.name)
            
        kernel_ax = Axis(f"kernel_{ax.name}", k)
        kernel_axes.append(kernel_ax)
        x = TargetedTensor(x, (curr_ax,)).unfold(kernel_ax, step=s)

    # 3. Target all kernel windows and input channels
    target_names = {ax.name for ax in kernel_axes} | {ax.name for ax in in_channel_axes}
    target_axes_in_x = tuple(ax for ax in x.topology if ax.name in target_names)

    # 4. Merge them into in_proj_dim
    in_proj_size = 1
    for ax in target_axes_in_x:
        in_proj_size *= ax.size
        
    in_proj_axis = Axis("in_proj_dim", in_proj_size)
    x_merged = TargetedTensor(x, target_axes_in_x).merge(in_proj_axis)

    # 5. Project to arbitrary output channel configuration
    return TargetedTensor(x_merged, (in_proj_axis,)).proj(*out_channel_axes, bias=bias, tie=tie_prefix)
