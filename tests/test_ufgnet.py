import torch

from degradations import build_degradation
from losses import UFGNetLoss
from models import UFGNet, project_hsi_to_msi


def _normalized_srf(msi_channels: int, hsi_channels: int) -> torch.Tensor:
    srf = torch.rand(msi_channels, hsi_channels)
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)


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
