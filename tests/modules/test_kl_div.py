# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import pytest
import torch
import torch.nn.functional as F

from fla.modules import FusedKLDivLoss
from fla.utils import assert_close, device, device_platform


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("T", [16, 32])
@pytest.mark.parametrize("D", [1024, 2048])
@pytest.mark.parametrize("V", [32000, 100000])
@pytest.mark.parametrize("reduction", ["batchmean"])
@pytest.mark.parametrize("accumulate_grad_in_fp32", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize(
    ('x_grad', 'weight_grad', 'context'),
    [(True, True, torch.enable_grad), (True, False, torch.enable_grad), (False, True, torch.enable_grad),
     (False, False, torch.enable_grad), (True, True, torch.no_grad), (True, True, torch.inference_mode)],
    ids=['both', 'input', 'weight', 'frozen', 'no_grad', 'inference'],
)
@pytest.mark.skipif(
    device_platform == 'intel',
    reason="Intel Triton Failure",
)
def test_fused(
    B: int,
    T: int,
    D: int,
    V: int,
    reduction: str,
    accumulate_grad_in_fp32: bool,
    dtype: torch.dtype,
    x_grad: bool,
    weight_grad: bool,
    context,
):
    torch.manual_seed(42)
    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_(x_grad)
    x_weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_(weight_grad)
    target_x = torch.randn(B * T, D).to(device).to(dtype=dtype)
    target_weight = torch.randn(V, D).to(device).to(dtype=dtype)

    if x_grad != weight_grad:
        x = x.detach().t().contiguous().t().requires_grad_(x_grad)
        x_weight = x_weight.detach().t().contiguous().t().requires_grad_(weight_grad)

    with context():
        ref = F.kl_div(
            F.linear(x, x_weight).log_softmax(-1),
            F.linear(target_x, target_weight).softmax(-1),
            reduction=reduction,
        ).to(dtype)
    do = torch.randn_like(ref).to(device)
    if ref.requires_grad:
        ref.backward(do)
    ref_dx, x.grad = x.grad, None
    ref_dw, x_weight.grad = x_weight.grad, None

    with context():
        tri = FusedKLDivLoss(
            reduction=reduction,
            accumulate_grad_in_fp32=accumulate_grad_in_fp32,
        )(x, target_x, x_weight, target_weight)
        if tri.requires_grad:
            dx, dw = tri.grad_fn.saved_tensors
            assert (dx is not None) == x_grad
            assert (dw is not None) == weight_grad
        tri = tri.to(dtype=dtype)
    assert tri.requires_grad == ref.requires_grad
    if tri.requires_grad:
        tri.backward(do)
    tri_dx, x.grad = x.grad, None
    tri_dw, x_weight.grad = x_weight.grad, None

    assert_close("  o", ref, tri, 1e-2)
    for name, expected, actual in [('dx', ref_dx, tri_dx), ('dw', ref_dw, tri_dw)]:
        if expected is None:
            assert actual is None
        else:
            assert_close(name, expected, actual, 1e-2)


def test_fused_rejects_trainable_teacher():
    torch.manual_seed(42)
    x = torch.randn(2, 4)
    weight = torch.randn(8, 4)
    target_x = torch.randn(2, 4, requires_grad=True)
    target_weight = torch.randn(8, 4)

    with pytest.raises(RuntimeError, match="frozen teacher"):
        FusedKLDivLoss()(x, target_x, weight, target_weight)

    target_x = target_x.detach()
    target_weight = target_weight.requires_grad_()
    with pytest.raises(RuntimeError, match="frozen teacher"):
        FusedKLDivLoss()(x, target_x, weight, target_weight)
