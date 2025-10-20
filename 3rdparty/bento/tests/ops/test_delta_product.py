# -*- coding: utf-8 -*-

from typing import List

import pytest
import torch
import torch.nn.functional as F

from fla.ops.gated_delta_product import chunk_gated_delta_product
from fla.ops.gated_delta_product.chunk_ref import chunk_gated_delta_product_ref
from fla.ops.gated_delta_product.naive import naive_recurrent_gated_delta_product
from fla.utils import assert_close, device


@pytest.mark.parametrize(
    ('B', 'T', 'H', 'D', 'scale', 'num_householder', 'use_qk_l2norm_in_kernel', 'dtype'),
    [
        pytest.param(
            *test,
            id="B{}-T{}-H{}-D{}-scale{}-num_householder{}-l2norm{}-{}".format(*test)
        )
        for test in [
            (1, 63, 1, 64, 0.1, 1, False, torch.float16),
            (2, 200, 3, 60, 0.1, 1, False, torch.float16),
            (2, 1000, 4, 64, 0.1, 2, False, torch.float16),
            (2, 1024, 4, 64, 1, 2, True, torch.float16),
            (2, 1024, 6, 100, 1, 2, False, torch.float16),
            (4, 1500, 8, 128, 0.1, 3, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, False, torch.float16),
            (2, 2048, 8, 128, 1, 3, True, torch.float16),
        ]
    ]
)
def test_chunk(
    B: int,
    T: int,
    H: int,
    D: int,
    scale: float,
    num_householder: int,
    use_qk_l2norm_in_kernel: bool,
    dtype: torch.dtype,
):
    torch.manual_seed(42)
    q = torch.randn(B, T, H, D, dtype=dtype)
    k = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    v = torch.randn(B, T * num_householder, H, D, dtype=dtype)
    beta = torch.rand(B, T * num_householder, H, dtype=dtype).sigmoid()
    h0 = torch.zeros(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(True), (q, k, v, beta, h0))

    tri, tri_ht = chunk_gated_delta_product(
        q=F.normalize(q.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else q.clone(),
        k=F.normalize(k.clone(), p=2, dim=-1) if not use_qk_l2norm_in_kernel else k.clone(),
        v=v.clone(),
        g=None,
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone(),
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )
    do = torch.randn_like(q)
    dht = torch.randn_like(h0)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=F.normalize(q.clone(), p=2, dim=-1),
        k=F.normalize(k.clone(), p=2, dim=-1),
        v=v.clone(),
        g=None,
        beta=beta.clone(),
        num_householder=num_householder,
        scale=scale,
        initial_state=h0.clone(),
        output_final_state=True,
    )

    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.008)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.008)
    assert_close('db', ref_dbeta, tri_dbeta, 0.02)
    assert_close('dh0', ref_dh0, tri_dh0, 0.008)


@pytest.mark.parametrize(
    ('H', 'D', 'num_householder', 'cu_seqlens', 'dtype'),
    [
        (2, 64, 3, [0, 63, ], torch.float16),
        (2, 100, 2, [0, 63, 100, 500, 1000], torch.float16),
        (2, 128, 2, [0, 100, 300, 800, 1500, 2000], torch.float16),
        (2, 256, 3, [0, 100, 123, 300, 500, 800, 1000, 1500, 2048], torch.float16),
    ]
)
def test_chunk_varlen(
    H: int,
    D: int,
    num_householder: int,
    cu_seqlens: List[int],
    dtype: torch.dtype,
):
    torch.manual_seed(42)

    T = cu_seqlens[-1]
    N = len(cu_seqlens) - 1
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)
    scale = 1.0

    q = torch.nn.functional.normalize(torch.randn((1, T, H, D), dtype=dtype), dim=-1, p=2)
    k = torch.nn.functional.normalize(torch.randn(1, T*num_householder, H, D, dtype=dtype), dim=-1, p=2)
    v = torch.randn((1, T*num_householder, H, D), dtype=dtype)
    beta = torch.rand(1, T*num_householder, H, dtype=dtype).sigmoid()
    h0 = torch.randn((N, H, D, D), dtype=dtype)

    q, k, v, beta, h0 = map(lambda x: x.to(device).requires_grad_(), (q, k, v, beta, h0))
    do = torch.randn_like(q)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_gated_delta_product(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=None,
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    ref, ref_ht = chunk_gated_delta_product_ref(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        beta=beta.clone(),
        g=None,
        scale=scale,
        output_final_state=True,
        num_householder=num_householder,
        initial_state=h0.clone(),
        cu_seqlens=cu_seqlens
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
    q.grad = k.grad = v.grad = beta.grad = h0.grad = None

    torch_ref = torch.zeros_like(ref)
    torch_ref_ht = torch.zeros_like(ref_ht)
    for i in range(len(cu_seqlens) - 1):
        start, end = cu_seqlens[i], cu_seqlens[i+1]
        q_i = q[:, start:end, :, :]
        k_i = k[:, start*num_householder:end*num_householder, :, :]
        v_i = v[:, start*num_householder:end*num_householder, :, :]
        beta_i = beta[:, start*num_householder:end*num_householder, :]
        o3_i, h3_i = naive_recurrent_gated_delta_product(
            q_i, k_i, v_i, None, beta_i, scale=scale, cu_seqlens=None, output_final_state=True, num_householder=num_householder
        )
        torch_ref[:, start:end, :, :] = o3_i
        torch_ref_ht[i, :, :, :] = h3_i.squeeze(0)

    ((torch_ref * do).sum() + (torch_ref_ht * dht).sum()).backward(retain_graph=True)

    assert_close('o', ref, tri, 0.005)
    assert_close('ht', ref_ht, tri_ht, 0.005)
    assert_close('dq', ref_dq, tri_dq, 0.007)
    assert_close('dk', ref_dk, tri_dk, 0.008)
    assert_close('dv', ref_dv, tri_dv, 0.007)
    assert_close('db', ref_dbeta, tri_dbeta, 0.015)
    assert_close('dh0', ref_dh0, tri_dh0, 0.007)
