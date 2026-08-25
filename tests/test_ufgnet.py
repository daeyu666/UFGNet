import torch

from degradations import build_degradation
from losses import SAMLoss, UFGNetLoss
from models import (
    FrequencyAwareSpectralAttention,
    UFGNet,
    project_hsi_to_msi,
)


def _normalized_srf(msi_channels: int, hsi_channels: int) -> torch.Tensor:
    srf = torch.rand(msi_channels, hsi_channels)
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)


def test_sam_matches_paper_equation_24():
    eps = 1e-8
    pred = torch.tensor([[[[1.0]], [[2.0]], [[3.0]]]])
    target = torch.tensor([[[[2.0]], [[1.0]], [[4.0]]]])
    loss = SAMLoss(eps=eps)(pred, target)

    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.linalg.vector_norm(pred, ord=2, dim=1)
    target_norm = torch.linalg.vector_norm(target, ord=2, dim=1)
    expected_cos = dot / (pred_norm * target_norm + eps)
    expected_cos = torch.clamp(expected_cos, -1.0 + eps, 1.0 - eps)
    expected = torch.acos(expected_cos).mean()

    assert torch.allclose(loss, expected, atol=1e-7, rtol=1e-7)


def test_fasa_matches_paper_equation_18():
    torch.manual_seed(3)
    residual = torch.tensor(
        [[[[1.0, 2.0]], [[0.5, -1.0]], [[2.0, 0.0]]]], dtype=torch.float32
    )
    guidance = torch.tensor(
        [[[[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]]]], dtype=torch.float32
    )
    tau = 2.0
    fasa = FrequencyAwareSpectralAttention(channels=3, tau=tau)
    _, affinity = fasa(residual, guidance)

    q = residual.reshape(1, 3, -1)
    correlation = torch.matmul(q, q.transpose(1, 2))
    f = guidance.mean(dim=(-2, -1))
    dist2 = (f.unsqueeze(2) - f.unsqueeze(1)).pow(2)
    physical_prior = torch.exp(-dist2 / tau)
    expected = torch.softmax(correlation * physical_prior, dim=-1)

    assert torch.allclose(affinity, expected, atol=1e-6, rtol=1e-6)


def test_ufgnet_forward_shapes_and_finite_values():
    torch.manual_seed(0)
    b, c, m, H, W, scale = 1, 8, 3, 16, 16, 4
    srf = _normalized_srf(m, c)
    degradation = build_degradation(
        "gaussian_bicubic", scale_ratio=scale, sigma=2.0, kernel_size=5
    )

    gt = torch.rand(b, c, H, W)
    lr_hsi = degradation.degrade(gt)
    hr_msi = project_hsi_to_msi(gt, srf)

    # K=3 is used in this small unit test because Eq. (20) explicitly gives
    # the 3x3 / 9-sample case; the experiment config defaults to paper K=7.
    model = UFGNet(
        hsi_channels=c,
        msi_channels=m,
        srf=srf,
        rank=5,
        qiem_regularization=1e-4,
        fasa_tau=1.0,
        deform_kernel_size=3,
    )
    pred, aux = model(lr_hsi, hr_msi, degradation, return_aux=True)

    assert pred.shape == gt.shape
    assert aux["initial_estimate"].shape == gt.shape
    assert aux["spectral_guidance"].shape == gt.shape
    assert aux["spatial_guidance"].shape == gt.shape
    assert aux["fasa_affinity"].shape == (b, c, c)
    assert aux["offset"].shape == (b, 18, H, W)
    assert aux["mask"].shape == (b, 9, H, W)
    assert torch.isfinite(pred).all()


def test_ufgnet_unsupervised_loss_backward():
    torch.manual_seed(1)
    b, c, m, H, W, scale = 1, 8, 3, 16, 16, 4
    srf = _normalized_srf(m, c)
    degradation = build_degradation(
        "gaussian_bicubic", scale_ratio=scale, sigma=2.0, kernel_size=5
    )

    gt = torch.rand(b, c, H, W)
    lr_hsi = degradation.degrade(gt)
    hr_msi = project_hsi_to_msi(gt, srf)

    model = UFGNet(c, m, srf, rank=5, deform_kernel_size=3)
    criterion = UFGNetLoss(
        degradation,
        srf,
        lambda_rec=1.0,
        lambda_sam=1e-2,
        lambda_freq=1e-2,
        gamma=0.5,
        eta=0.5,
    )

    pred = model(lr_hsi, hr_msi, degradation)
    loss, parts = criterion(pred, lr_hsi, hr_msi)
    loss.backward()

    assert torch.isfinite(loss)
    for value in parts.values():
        assert torch.isfinite(value)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() for g in grads)
