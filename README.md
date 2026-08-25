# UFGNet / HSI Super-Resolution Experiments

当前仓库用于 HSI-MSI Fusion 超分复现与对比实验。仓库保留 3 种 LR-HSI 退化方式，并在 `EMR-Diff/` 中加入 EMR-Diff 对比实现。

## 当前统一对比协议

当前阶段先只使用常规退化，不使用 physical 退化：

```text
scale_ratio = 4
degradation_mode = gaussian_bicubic
degradation_kernel_size = 5
degradation_sigma = 2.0
n_select_bands = 8
msi_mode = uniform
```

即：

```text
HR-HSI
  -> 5x5 Gaussian blur, sigma=2
  -> Bicubic downsampling x4
  -> LR-HSI
```

PaviaU、Houston13、Chikusei 当前均使用 8-band HR-MSI。

数据设置：

```text
train patch = 64x64
stride = 32
test size = 128x128
```

## 数据

原始 HSI 放在：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

## 主要文件

| 文件/目录 | 说明 |
|---|---|
| `config.py` | 当前统一实验配置 |
| `data_loader.py` | HSI 读取、归一化、patch、LR-HSI 与 HR-MSI 构建 |
| `degradations/` | Bicubic、Gaussian+Bicubic、Physical 三种退化 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 相关工具，后续真实 SRF 实验继续保留 |
| `EMR-Diff/` | EMR-Diff 开源代码及 UFGNet 协议适配 |

## 3 种 LR-HSI 退化

仓库仍保留：

- `bicubic`：纯 Bicubic 下采样；
- `gaussian_bicubic`：Gaussian blur + Bicubic；
- `physical`：Gaussian PSF/MTF + detector area averaging + sampling。

当前对比实验固定使用 `gaussian_bicubic`。后续切换其他退化时再单独进行对应实验。

## 检查当前数据协议

```bash
python main.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --degradation_sigma 2.0 \
  --degradation_kernel_size 5 \
  --scale_ratio 4 \
  --n_select_bands 8 \
  --msi_mode uniform
```

Chikusei：

```bash
python main.py \
  --dataset Chikusei \
  --degradation_mode gaussian_bicubic \
  --degradation_sigma 2.0 \
  --degradation_kernel_size 5 \
  --scale_ratio 4 \
  --n_select_bands 8 \
  --msi_mode uniform
```

## EMR-Diff 对比实验

EMR-Diff 不再使用其原始固定 `256x256 / x8 / 31+3 channels` 数据协议。当前比较入口直接复用根目录 UFGNet 数据管线，使 LR-HSI、HR-MSI、GT、归一化、训练 patch 和测试区域一致。

训练 PaviaU：

```bash
cd EMR-Diff
python Train.py --dataset PaviaU
```

训练 Chikusei：

```bash
cd EMR-Diff
python Train.py --dataset Chikusei
```

快速单轮检查：

```bash
python Train.py --dataset PaviaU --epochs 1 --test_frequency 1
```

测试：

```bash
python Test.py --dataset PaviaU
python Test.py --dataset Chikusei
```

EMR-Diff 的状态通道按数据集自动计算：

```text
state_channels = HSI_channels + 8
PaviaU   = 103 + 8 = 111
Chikusei = 128 + 8 = 136
```

模型内部原先写死的 34 通道分组卷积、GroupNorm、`cuda:1`、x8 上采样以及 Harvard 路径均已改为动态配置或数据尺寸驱动。

## 物理退化与渐进退化

Physical 与 progressive degradation 代码仍保留，用于后续独立实验，不参与当前 EMR-Diff 常规退化对比。原有 trajectory sanity check、终点闭合检查与相关测试也继续保留。
