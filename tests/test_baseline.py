import torch

from baseline import baseline


def test_baseline_supports_103_hsi_4_msi_contract():
    model = baseline(
        scale_ratio=4,
        n_select_bands=4,
        n_bands=103,
        channels=16,
        num_blocks=2,
    )
    lr_hsi = torch.rand(1, 103, 8, 8)
    hr_msi = torch.rand(1, 4, 32, 32)
    pred, aux = model(lr_hsi, hr_msi)
    assert pred.shape == (1, 103, 32, 32)
    assert aux.shape == pred.shape
    assert torch.isfinite(pred).all()
