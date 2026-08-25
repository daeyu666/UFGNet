import torch

from degradations import build_degradation
from models import project_hsi_to_msi
from ufgnet_data import UFGNetFullSceneDataset, UFGNetPatchDataset


def _normalized_srf(msi_channels: int, hsi_channels: int) -> torch.Tensor:
    torch.manual_seed(17)
    srf = torch.rand(msi_channels, hsi_channels)
    return srf / srf.sum(dim=1, keepdim=True).clamp_min(1e-8)


def test_patches_are_sliced_from_one_predegraded_scene_pair():
    torch.manual_seed(5)
    c, m, h, w, scale = 8, 3, 64, 80, 4
    gt = torch.rand(c, h, w)
    srf = _normalized_srf(m, c)
    degradation = build_degradation(
        "gaussian_bicubic", scale_ratio=scale, sigma=2.0, kernel_size=5
    )

    lr_full = degradation.degrade(gt.unsqueeze(0)).squeeze(0)
    hr_msi_full = project_hsi_to_msi(gt.unsqueeze(0), srf).squeeze(0)

    dataset = UFGNetPatchDataset(
        gt_full=gt,
        hr_msi_full=hr_msi_full,
        lr_hsi_full=lr_full,
        patch_size=32,
        stride=16,
        scale_ratio=scale,
        augment=False,
    )

    # 64 -> positions 0,16,32; 80 -> 0,16,32,48.
    assert len(dataset) == 12
    index = dataset.coords.index((16, 32))
    sample = dataset[index]

    assert torch.equal(sample["gt"], gt[:, 16:48, 32:64])
    assert torch.equal(sample["hr_msi"], hr_msi_full[:, 16:48, 32:64])
    assert torch.equal(sample["lr_hsi"], lr_full[:, 4:12, 8:16])


def test_full_scene_dataset_returns_entire_observation_pair():
    torch.manual_seed(6)
    c, m, h, w, scale = 8, 3, 64, 80, 4
    gt = torch.rand(c, h, w)
    srf = _normalized_srf(m, c)
    degradation = build_degradation(
        "gaussian_bicubic", scale_ratio=scale, sigma=2.0, kernel_size=5
    )
    lr_full = degradation.degrade(gt.unsqueeze(0)).squeeze(0)
    hr_msi_full = project_hsi_to_msi(gt.unsqueeze(0), srf).squeeze(0)

    dataset = UFGNetFullSceneDataset(gt, hr_msi_full, lr_full)
    sample = dataset[0]

    assert len(dataset) == 1
    assert sample["gt"].shape == (c, h, w)
    assert sample["hr_msi"].shape == (m, h, w)
    assert sample["lr_hsi"].shape == (c, h // scale, w // scale)
    assert torch.equal(sample["gt"], gt)
    assert torch.equal(sample["hr_msi"], hr_msi_full)
    assert torch.equal(sample["lr_hsi"], lr_full)
