# UFGNet / HSI Super-Resolution Experiments

当前仓库用于 HSI-MSI Fusion 超分复现与对比实验。主线包含 UFGNet 复现、EMR-Diff 对比实现以及 3 种 LR-HSI 退化方式。

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

## SRF 物理覆盖保护

`wv2_all8` 现在表示“以 WV2 全部 8 个真实波段作为候选集合”，并不表示无条件生成 8 通道 MSI。代码会先在归一化之前计算每个真实 SRF 在当前 HSI 光谱支持范围内的完整能量覆盖率：

```math
rho_m = integral_{HSI support} S_m(lambda) d lambda / integral S_m(lambda) d lambda
```

默认阈值为 `0.90`，低于阈值的波段在离散化和归一化之前被剔除，避免把极小的截断尾部重新归一化成完整 MSI 通道。

当前 WV2 审计结果：

```text
PaviaU    -> 5/8: Blue, Green, Yellow, Red, RedEdge
Houston13 -> 8/8
Chikusei  -> 8/8
```

可独立检查：

```bash
python audit_srf_coverage.py
```

`wv2_visible5`、`wv2_visible6` 和 `wv2_all8` 仍保留为候选集合选项；`srf_coverage_policy=filter` 为当前正式物理保护模式。

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
| `EMR-Diff/` | EMR-Diff 对比实现与协议适配 |

## EMR-Diff 对比实验

`EMR-Diff/` 目录保留原有对比实现。当前主仓库的 UFGNet SRF 协议已经切换为真实 SRF + 覆盖保护；后续若要求 EMR-Diff 与 UFGNet 做正式同表比较，应让 EMR-Diff 也读取同一个根目录观测对，而不是继续使用历史的 uniform 8-band 观测。

训练入口仍可使用：

```bash
cd EMR-Diff
python Train.py --dataset PaviaU
python Train.py --dataset Chikusei
```

## 3 种 LR-HSI 退化

仓库保留：

- `bicubic`：纯 Bicubic 下采样；
- `gaussian_bicubic`：Gaussian blur + Bicubic；
- `physical`：Gaussian PSF/MTF + detector area averaging + sampling。

当前 UFGNet 首阶段正式基线固定使用 `gaussian_bicubic`；Physical 与 progressive degradation 保留为后续独立扩展。
