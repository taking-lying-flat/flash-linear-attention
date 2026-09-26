# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

from unittest.mock import create_autospec

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
):
    torch.manual_seed(42)
    x = torch.randn(B * T, D).to(device).to(dtype=dtype).requires_grad_()
    x_weight = torch.randn(V, D).to(device).to(dtype=dtype).requires_grad_()
    target_x = torch.randn(B * T, D).to(device).to(dtype=dtype)
    target_weight = torch.randn(V, D).to(device).to(dtype=dtype)

    ref = F.kl_div(
        F.linear(x, x_weight).log_softmax(-1),
        F.linear(target_x, target_weight).softmax(-1),
        reduction=reduction,
    ).to(dtype)
    do = torch.randn_like(ref).to(device)
    ref.backward(do)
    ref_dx, x.grad = x.grad.clone(), None
    ref_dw, x_weight.grad = x_weight.grad.clone(), None

    tri = FusedKLDivLoss(
        reduction=reduction,
        accumulate_grad_in_fp32=accumulate_grad_in_fp32,
    )(x, target_x, x_weight, target_weight).to(dtype=dtype)
    tri.backward(do)
    tri_dx, x.grad = x.grad.clone(), None
    tri_dw, x_weight.grad = x_weight.grad.clone(), None

    assert_close("  o", ref, tri, 1e-2)
    assert_close(" dx", ref_dx, tri_dx, 1e-2)
    assert_close(" dw", ref_dw, tri_dw, 1e-2)


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


@pytest.mark.parametrize(('use_dx', 'use_dw'), [(True, True), (True, False), (False, True), (False, False)])
def test_fused_ascend_dispatch(monkeypatch, use_dx, use_dw):
    from fla.modules.backends import modules_registry
    from fla.modules.backends.triton_ascend import TritonAscendBackend
    from fla.modules.backends.triton_ascend import fused_kl_div as npu
    from fla.modules.fused_kl_div import fused_kl_div_bwd, fused_kl_div_fwd
    from fla.ops.backends import _DISPATCH_DISABLED

    if _DISPATCH_DISABLED:
        pytest.skip('backend dispatch is disabled')

    backend = TritonAscendBackend()
    monkeypatch.setattr(backend, 'is_available', lambda: True)
    monkeypatch.setattr(modules_registry, '_get_sorted_backends', lambda: [backend])
    inputs = dict(x=object(), target_x=object(), weight=object(), target_weight=object())
    loss, dx, dw = object(), object() if use_dx else None, object() if use_dw else None
    fwd = create_autospec(npu.fused_kl_div_fwd_npu, return_value=(loss, dx, dw))
    bwd = create_autospec(npu.fused_kl_div_bwd_npu, return_value=(dx, dw))
    monkeypatch.setattr(npu, 'fused_kl_div_fwd_npu', fwd)
    monkeypatch.setattr(npu, 'fused_kl_div_bwd_npu', bwd)

    assert fused_kl_div_fwd(**inputs, use_dx=use_dx, use_dw=use_dw) == (loss, dx, dw)
    fwd.assert_called_once_with(
        **inputs,
        reduction='batchmean',
        accumulate_grad_in_fp32=True,
        use_dx=use_dx,
        use_dw=use_dw,
    )
    do = object()
    assert fused_kl_div_bwd(do=do, dx=dx, dw=dw) == (dx, dw)
    bwd.assert_called_once_with(do=do, dx=dx, dw=dw)
