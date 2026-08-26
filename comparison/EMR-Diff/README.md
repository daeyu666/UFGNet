# EMR-Diff Comparison

本目录保存 EMR-Diff 对比实验代码及其 UFGNet 协议适配。EMR-Diff 已从仓库根目录迁移到：

```text
comparison/EMR-Diff/
```

后续与 EMR-Diff 有关的训练代码、配置、模型权重和实验结果均保持在本目录内，避免与 UFGNet 主实验及其他对比方法混放。

## Shared comparison protocol

EMR-Diff 通过 `dataset_loader/ufg_adapter.py` 复用仓库根目录的 UFGNet 数据协议，因此 LR-HSI、HR-MSI、GT、训练 patch、完整场景评价、退化算子、SRF 规则和评价指标与当前 UFGNet 主实验保持一致。

当前 LR-HSI 基线使用：

```text
scale factor = x4
5x5 Gaussian blur, sigma = 2
+ bicubic downsampling
```

MSI/SRF 的具体通道配置跟随仓库根目录当前 UFGNet 配置，不在 EMR-Diff 内单独重新生成一套观测协议。

原始 EMR-Diff `dataset_loader/dataloader.py` 继续保留作为开源代码参考，但正式对比入口使用 `dataset_loader/ufg_adapter.py`。

## Data

原始数据仍统一放在仓库根目录：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

移动到 `comparison/EMR-Diff/` 后，配置中的相对数据路径已经同步调整。

## Train

可以从仓库根目录直接运行：

```bash
python comparison/EMR-Diff/Train.py --dataset PaviaU
python comparison/EMR-Diff/Train.py --dataset Houston13
python comparison/EMR-Diff/Train.py --dataset Chikusei
```

也可以进入本目录运行：

```bash
cd comparison/EMR-Diff
python Train.py --dataset PaviaU
```

快速检查：

```bash
python Train.py --dataset PaviaU --epochs 1 --test_frequency 1
```

## Test

默认加载该数据集最新 checkpoint：

```bash
python Test.py --dataset PaviaU
python Test.py --dataset Chikusei
```

指定 checkpoint：

```bash
python Test.py \
  --dataset PaviaU \
  --checkpoint checkpoints/EMRDIFF_PaviaU/model_epoch_100.pth.tar
```

## Experiment storage rule

EMR-Diff 的模型权重只保存在本方法目录下：

```text
comparison/EMR-Diff/checkpoints/
└── EMRDIFF_<dataset>/
    └── model_epoch_<N>.pth.tar
```

EMR-Diff 的测试结果、重建结果和其他实验输出只保存在：

```text
comparison/EMR-Diff/outputs/
└── <dataset>/
```

`checkpoints/` 与 `outputs/` 的目录占位符保留在 Git 中，但其中生成的权重和实验结果默认被 `.gitignore` 忽略。

后续若增加新的对比方法，应在 `comparison/<Method>/` 下建立独立目录，并采用相同的 `checkpoints/`、`outputs/` 隔离规则。统一规范见 `comparison/README.md`。
