"""Parameter/source audit for the UFGNet reproduction.

This script is diagnostic only.  It does not force the implementation to match
paper Table III by changing undocumented channel widths or grouping rules.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import torch

from models import UFGNet


@dataclass(frozen=True)
class PaperCase:
    name: str
    hsi_channels: int
    msi_channels: int
    reported_params_m: float


PAPER_CASES = (
    PaperCase("Pavia", 93, 4, 0.198),
    PaperCase("CAVE", 31, 3, 0.072),
    PaperCase("Botswana", 145, 5, 0.383),
)


def normalized_srf(msi_channels: int, hsi_channels: int) -> torch.Tensor:
    x = torch.ones(msi_channels, hsi_channels, dtype=torch.float32)
    return x / x.sum(dim=1, keepdim=True)


def count_trainable(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build_case(hsi_channels: int, msi_channels: int, kernel_size: int) -> UFGNet:
    return UFGNet(
        hsi_channels=hsi_channels,
        msi_channels=msi_channels,
        srf=normalized_srf(msi_channels, hsi_channels),
        rank=min(5, hsi_channels),
        deform_kernel_size=kernel_size,
    )


def print_model_breakdown(model: UFGNet) -> None:
    total = count_trainable(model)
    print(f"  total: {total:,} ({total / 1e6:.6f} M)")
    for name in ("qiem", "fgm", "ccrm"):
        module = getattr(model, name)
        n = count_trainable(module)
        print(f"  {name:>5}: {n:,} ({n / 1e6:.6f} M)")


def audit_paper_cases() -> None:
    print("Paper Table III parameter audit")
    print("The paper reports r=5 and sensitivity-optimal CCRM kernel size K=7,")
    print("while Eq. (20) only uses 3x3 as an illustrative sampling example.\n")
    for case in PAPER_CASES:
        print(f"[{case.name}] Ch={case.hsi_channels}, Cm={case.msi_channels}")
        for k in (3, 7):
            model = build_case(case.hsi_channels, case.msi_channels, k)
            total = count_trainable(model)
            delta = total / 1e6 - case.reported_params_m
            print(
                f"  K={k}: {total / 1e6:.6f} M; "
                f"paper={case.reported_params_m:.3f} M; delta={delta:+.6f} M"
            )
        print()

    print(
        "Interpretation: a mismatch is expected unless the paper's omitted "
        "implementation details (e.g. deformable-convolution grouping/channel "
        "mixing or hidden feature widths) are known. Do not alter the published "
        "equations merely to force Table III agreement."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hsi_channels", type=int, default=0)
    parser.add_argument("--msi_channels", type=int, default=0)
    parser.add_argument("--kernel_size", type=int, default=7)
    args = parser.parse_args()

    audit_paper_cases()
    if args.hsi_channels > 0 and args.msi_channels > 0:
        print("\nCustom configuration")
        model = build_case(args.hsi_channels, args.msi_channels, args.kernel_size)
        print_model_breakdown(model)


if __name__ == "__main__":
    main()
