# EMR-Diff

This directory contains the EMR-Diff open-source implementation adapted to run under the shared UFGNet comparison protocol in this repository.

## Current comparison protocol

The current baseline comparison uses the same data pipeline as the repository root:

- datasets: `PaviaU`, `Houston13`, `Chikusei`
- scale factor: `x4`
- LR-HSI: `5x5 Gaussian (sigma=2) + bicubic downsampling`
- HR-MSI: `8 uniformly selected HSI bands`
- training patch: `64x64`
- stride: `32`
- test region: center `128x128`
- normalization and sample construction: shared with root `data_loader.py`
- metrics: shared with root `metrics.py`

The original EMR-Diff dataset loader is retained in `dataset_loader/dataloader.py` for reference, but the comparison entrypoints use `dataset_loader/ufg_adapter.py` so EMR-Diff and UFGNet receive the same generated LR-HSI / HR-MSI / GT samples.

## Data

Place the raw HSI files in the repository root:

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

## Train

Run from the `EMR-Diff` directory or from the repository root.

```bash
python Train.py --dataset PaviaU
python Train.py --dataset Chikusei
```

Optional quick run:

```bash
python Train.py --dataset PaviaU --epochs 1 --test_frequency 1
```

`Houston13` is also accepted by the same entrypoint:

```bash
python Train.py --dataset Houston13
```

## Test

By default `Test.py` loads the latest dataset-specific checkpoint:

```bash
python Test.py --dataset PaviaU
python Test.py --dataset Chikusei
```

A specific checkpoint can be supplied with:

```bash
python Test.py --dataset PaviaU --checkpoint checkpoints/EMRDIFF_PaviaU/model_epoch_100.pth.tar
```

## Dynamic channels

The original code was hard-coded for `31 HSI + 3 MSI = 34` channels. The comparison version keeps the original dense + depthwise grouping idea but derives channels from each dataset:

```text
state_channels = HSI_channels + 8
```

For the two current experiments this gives:

```text
PaviaU:   103 + 8 = 111
Chikusei: 128 + 8 = 136
```

The pseudo-MSI branch follows the original code's use of the first HSI bands and is generalized from the first 3 bands to the first 8 bands. This avoids copying the HR-MSI directly into the pseudo-MSI state and preserves a non-trivial multimodal residual.

## Outputs

Checkpoints:

```text
EMR-Diff/checkpoints/EMRDIFF_<dataset>/
```

Predictions:

```text
EMR-Diff/outputs/<dataset>/
```
