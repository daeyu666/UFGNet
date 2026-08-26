# UFGNet / HSI Super-Resolution Experiments

当前仓库专门用于 UFGNet 的 HSI-MSI Fusion 超分复现与主线实验，并保留 3 种 LR-HSI 退化方式。对比实验已迁移到独立仓库 `daeyu666/comparison_experiments`，不再把其他模型、权重和对比结果放在本仓库中。

## 当前 UFGNet 复现协议

首阶段固定为：

```text
scale_ratio = 4
degradation_mode = gaussian_bicubic
degradation_kernel_size = 5
degradation_sigma = 2.0
patch_size = 64
stride = 32
batch_size = 1
optimizer = Adam
lr = 5e-4
msi_mode = srf
srf_band_set = auto
```

LR-HSI 由完整 HR-HSI 场景先执行 `5x5 Gaussian(sigma=2) + Bicubic x4` 生成一次，再从同一观测对切取重叠 patch；最终指标在完整场景上评估。数据生成、CCRM 残差回投和无监督 loss 共用同一个空间退化算子与同一 SRF。

UFGNet 完整实现 QIEM、FGM、CCRM、SpeDOB、SpaDOB、FASA，不做结构简化；损失包含 reconstruction、SAM、integrated dual-domain frequency consistency；评价指标为 PSNR、SSIM、ERGAS、SAM、CC、RMSE。

训练入口：

```bash
python train_ufgnet.py \
  --dataset PaviaU \
  --scale_ratio 4 \
  --degradation_mode gaussian_bicubic \
  --degradation_sigma 2.0 \
  --degradation_kernel_size 5 \
  --msi_mode srf \
  --patch_size 64 \
  --stride 32 \
  --batch_size 1 \
  --lr 5e-4 \
  --device cuda
```

## 数据集对应真实 SRF

默认 `srf_band_set=auto` 不再强制所有数据集使用同一组传感器波段，而是在每个数据集内固定一个物理一致的观测模型：

```text
PaviaU    -> IKONOS Blue / Green / Red / NIR (4 bands)
Houston13 -> WorldView-2 all8
Chikusei  -> WorldView-2 all8
```

PaviaU 的 IKONOS 数值曲线保存在 `data/srf/ikonos_relative_spectral_response.csv`，由 HySure 项目公开的 `ikonos_spec_resp.mat` 数值数组转换而来；来源、原始 blob SHA 与验证说明见 `data/srf/IKONOS_SOURCE.md`。

PaviaU 的公开基准通常描述为 103 个有效波段覆盖 430-860 nm。原仓库 `PaviaU.txt` 的 430-838 nm 网格继续保留用于旧实验复核；当前 IKONOS 主协议使用 `PaviaU_nominal_430_860.txt`，对应常见 Pavia/ROSIS fusion benchmark 的 430-860 nm 映射约定。

## SRF 物理覆盖保护

无论使用 IKONOS 还是 WV2，代码都会在归一化之前计算每个真实 SRF 在当前 HSI 光谱支持范围内的完整能量覆盖率：

```math
rho_m = integral_{HSI support} S_m(lambda) d lambda / integral S_m(lambda) d lambda
```

默认阈值为 `0.90`，低于阈值的波段在离散化和归一化之前被剔除，避免把极小的截断尾部重新归一化成完整 MSI 通道。

当前默认审计预期：

```text
PaviaU / IKONOS4
  Blue  > 96%
  Green > 98%
  Red   > 99%
  NIR   > 92%
  => 4/4 retained

Houston13 / WV2 all8
  => 8/8 retained

Chikusei / WV2 all8
  => 8/8 retained
```

可独立检查：

```bash
python audit_srf_coverage.py
```

`ikonos4`、`wv2_visible5`、`wv2_visible6` 和 `wv2_all8` 均可通过 `--srf_band_set` 显式指定；`auto` 为正式默认。`srf_coverage_policy=filter` 为当前物理保护模式。

## UFGNet 来源一致性说明

1. FGM 空间分支按 Algorithm 1 / Eq. (13) 的 phase-only inverse FFT + 3x3 `C_phase` 实现；Fig. 4 中额外 cosine/sine 示意不静默补全。
2. CCRM 中符号 `K` 在论文中存在不兼容定义；当前保留方法部分明确的 3x3 deformable convolution，不把敏感性中的 `K=7` 擅自解释成 7x7 DConv。
3. FASA Eq. (18) 使用数值稳定的等价形式 `softmax(QQ^T - distance^2/tau)`。
4. SAM Eq. (24) 的 epsilon 放在两项 L2 范数乘积之后。
5. PSNR、SSIM、ERGAS、SAM、CC、RMSE 对网络原始输出计算，不在指标前静默 clamp 到 `[0,1]`。
6. 论文未披露 QIEM 正则系数和 FASA 温度的数值，代码保留为显式可调参数。

## Sanity / audit

```bash
python sanity_check_ufgnet.py --dataset PaviaU --scale_ratio 4 --degradation_mode gaussian_bicubic --msi_mode srf --device cuda
python smoke_train_ufgnet.py --dataset PaviaU --scale_ratio 4 --degradation_mode gaussian_bicubic --msi_mode srf --device cuda --epochs 20
python audit_ufgnet.py
python audit_srf_coverage.py
```

CPU CI 检查退化闭合、SRF 覆盖、完整 UFGNet 前向、loss backward、FASA、SAM 以及单场景预退化 patch 协议。

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
| `config.py` | 数据、SRF、退化、UFGNet 与训练配置 |
| `data_loader.py` | HSI 读取、归一化、LR-HSI / HR-MSI 构建 |
| `ufgnet_data.py` | UFGNet 单场景预退化 + 重叠 patch 协议 |
| `degradations/` | Bicubic、Gaussian+Bicubic、Physical 三种退化 |
| `models/ufgnet.py` | UFGNet：QIEM、FGM、CCRM、SpeDOB、SpaDOB、FASA |
| `train_ufgnet.py` | UFGNet 无监督训练与完整场景评价 |
| `losses.py` | UFGNet 三项无监督复合损失 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | 真实 SRF 插值、覆盖校验、离散积分权重 |
| `audit_srf_coverage.py` | 数据集与真实 SRF 物理覆盖审计 |
| `audit_ufgnet.py` | 论文参数量与结构歧义审计 |

## 对比实验已迁移

所有对比方法统一放入独立仓库：

```text
daeyu666/comparison_experiments
└── comparison/
    ├── EMR-Diff/
    │   ├── checkpoints/
    │   ├── outputs/
    │   └── logs/
    └── <OtherMethod>/
```

后续新增对比方法直接在 `comparison_experiments/comparison/<Method>/` 下建立独立目录，每个方法的模型权重、实验结果和日志均保存在自己的方法目录内。UFGNet 仓库不再保存对比方法实现，避免主线代码继续膨胀。

## 3 种 LR-HSI 退化

仓库保留：

- `bicubic`：纯 Bicubic 下采样；
- `gaussian_bicubic`：Gaussian blur + Bicubic；
- `physical`：Gaussian PSF/MTF + detector area averaging + sampling。

当前 UFGNet 首阶段正式基线固定使用 `gaussian_bicubic`；Physical 与 progressive degradation 保留为后续独立扩展。
