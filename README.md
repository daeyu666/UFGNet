# S2Diff: HSI Super-Resolution Experimental Scaffold

当前仓库用于 HSI-MSI Fusion 超分实验。基础代码提供数据读取、SRF、指标、可切换 LR-HSI 退化和渐进物理退化轨迹；具体扩散网络尚未接入。

## UFGNet reproduction

当前 `reproduce-ufgnet` 分支用于复现论文 UFGNet，实验协议固定为：

- 4× HSI-MSI fusion；
- LR-HSI 首阶段采用仓库默认 Gaussian blur + Bicubic 下采样；
- HR-MSI 使用仓库现有 SRF；
- 网络完整实现 QIEM、FGM、CCRM、SpeDOB、SpaDOB、FASA，不做结构简化；
- 无监督损失完整实现 reconstruction、SAM、integrated dual-domain frequency consistency；
- 评价指标为 PSNR、SSIM、ERGAS、SAM、CC、RMSE；
- 数据生成、CCRM 残差回投与 loss 共用同一个空间退化算子和同一 SRF。

当前实现入口：

```bash
python train_ufgnet.py \
  --dataset PaviaU \
  --scale_ratio 4 \
  --degradation_mode gaussian_bicubic \
  --msi_mode srf
```

论文明确支持并已写入配置的参数包括：`r=5`、`lambda_rec=1`、`lambda_sam=1e-2`、`lambda_freq=1e-2`、`gamma=0.5`、`eta=0.5`、Adam 初始学习率 `5e-4` 与 batch size `1`。论文未给出 QIEM 正则系数 λ 与 FASA 温度 τ 的具体数值，因此代码将二者保留为显式可调参数 `ufg_qiem_regularization` 与 `ufg_fasa_tau`，避免把未披露数值伪装成论文设定。

### Source-faithfulness notes

当前复现对论文中存在的歧义不做静默补全：

1. FGM 空间分支：Fig. 4 的示意图包含 `cos(phi)` / `sin(phi)` 卷积分支与 `atan2`，但 Algorithm 1 和 Eq. (13) 明确定义为 phase-only inverse FFT 后接 3×3 `C_phase`。当前可执行实现以 Algorithm 1 / Eq. (13) 为准，并在代码中保留该差异说明；
2. CCRM 中符号 `K` 被论文重复用于不兼容的含义。Eq. (20) 明确定义 `K` 为**采样元素总数**，并给出“3×3 kernel 时 K=9”；参数敏感性部分又将 `K` 称为**kernel size**，讨论 K=3、7、9 并报告 K=7 最优。由于该表述无法唯一映射为标准二维 deformable-convolution kernel，当前正式配置保留方法部分明确画出的 3×3 DConv，而不擅自把敏感性中的 K=7 解释成 7×7 DConv；
3. Table III 参数量无法由论文已披露结构唯一复原。当前标准 full-channel 3×3 deformable convolution 在 Pavia 上约 0.130M，而论文报告 0.198M；若将敏感性 K=7 直接解释成 7×7 DConv，则会上升到约 0.577M，反而明显偏离论文参数量。仓库保留 `audit_ufgnet.py` 显式报告该差异，不通过猜测 grouped/depthwise 结构强行凑参数量；
4. FASA Eq. (18) 已按公式实现为 `softmax((QQ^T) * P)`，其中 `P_ij=exp(-||f_i-f_j||^2/tau)`；SAM Eq. (24) 的 epsilon 仅加在两项 L2 范数乘积之后。

### Reproduction sanity checks

CPU CI 当前检查结果：

```text
12 tests passed
```

覆盖退化终点闭合、数据端退化一致性、完整 UFGNet 前向、无监督 loss backward、FASA Eq. (18)、SAM Eq. (24) 等。

真实 HSI 长训练前先运行：

```bash
python sanity_check_ufgnet.py \
  --dataset PaviaU \
  --scale_ratio 4 \
  --degradation_mode gaussian_bicubic \
  --msi_mode srf \
  --device cuda
```

该脚本执行 1 个真实训练 batch 的完整前向/反向，并打印 QIEM、FGM、CCRM 中间量、FASA 行和、offset/mask、各损失项、各模块梯度范数、参数量以及 6 个 GT 监控指标。GT 仅用于监控，不进入无监督训练 loss。

参数量审计：

```bash
python audit_ufgnet.py
```

依赖列表见 `requirements-ufgnet.txt`。实际 GPU 训练建议安装与 CUDA 匹配的 `torchvision`，优先使用其原生 modulated deformable convolution；仓库同时保留纯 PyTorch bilinear `grid_sample` fallback 用于可移植性检查。

## 当前代码

| 文件/目录 | 说明 |
|---|---|
| `data_loader.py` | HSI 读取、patch 构建、LR-HSI 退化、HR-MSI 构建、DataLoader |
| `degradations/` | Bicubic、Gaussian+Bicubic、Physical 退化及 progressive trajectory |
| `check_degradation_trajectory.py` | 在接网络前检查各时间步退化状态 |
| `models/ufgnet.py` | UFGNet：QIEM、FGM、CCRM、SpeDOB、SpaDOB、FASA |
| `train_ufgnet.py` | UFGNet 无监督训练与全参考指标监控 |
| `sanity_check_ufgnet.py` | 真实数据单 batch 前向/反向诊断 |
| `audit_ufgnet.py` | 论文 Table III 参数量对照与结构歧义审计 |
| `losses.py` | SAMLoss 与 UFGNet 三项无监督复合损失 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、离散积分权重、HSI→MSI |
| `config.py` | 数据、退化、UFGNet 与训练配置 |
| `requirements-ufgnet.txt` | UFGNet 复现依赖 |
| `main.py` | 检查数据与退化配置是否正确串联 |

## Degradation v1

### 1. LR-HSI 退化模式

三种退化通过 `--degradation_mode` 切换：

- `bicubic`：纯 Bicubic 下采样；
- `gaussian_bicubic`：Gaussian blur + Bicubic，下采样基线；
- `physical`：Gaussian PSF/MTF + detector area averaging + sampling，作为真实物理退化扩展。

UFGNet 首阶段复现默认：

```text
degradation_mode = gaussian_bicubic
scale_ratio = 4
degradation_sigma = 2.0
degradation_kernel_size = 5
msi_mode = srf
```

旧 `make_lr_hsi()` 在未传入退化算子时仍保持 Gaussian+Bicubic 概念基线，避免旧脚本静默改变结果。

### 2. Progressive degradation

默认：

```text
T = 12
scale stages = 1 -> 2 -> 4
lift = auto
```

`auto` 会解析为：

- `physical` → `normalized_adjoint`；
- `bicubic / gaussian_bicubic` → `bilinear`。

物理模式下：

```math
D_t = B_{r_t} P_t
```

```math
\tilde D_t = U_t D_t
```

逆过程预留更新：

```math
x_{t-1}=x_t+\tilde D_{t-1}(\hat X_0)-\tilde D_t(\hat X_0)
```

所有 diffusion state 均位于 HR 网格。

### 3. 数据端与扩散终点闭合

`build_datasets()` 只构建一个 `degradation_operator`，训练集、测试集和 `ProgressiveDegradation` 共用该对象。

必须满足：

```math
D_T(X)=Y_{LR-HSI}
```

对应测试位于：

```text
tests/test_degradation_closure.py
tests/test_data_pipeline_degradation.py
```

## 退化轨迹 sanity check

先把原始 HSI 放入：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

### Gaussian+Bicubic UFGNet 基线

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode gaussian_bicubic \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12 \
  --legacy_sigma 2.0 \
  --legacy_kernel 5
```

### Physical 后续扩展

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode physical \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12 \
  --mtf_nyquist 0.2
```

### Bicubic 对照

```bash
python check_degradation_trajectory.py \
  --dataset PaviaU \
  --mode bicubic \
  --lift_mode auto \
  --crop_size 128 \
  --scale_ratio 4 \
  --total_steps 12
```

脚本逐时间步输出：

```text
t, scale, strength, scale_transition, step_l1,
PSNR, SAM(deg), mean, std, HF ratio
```

重点先检查：

1. `terminal_closure_max_abs_error` 是否接近 0；
2. `mean` 是否在 1→2→4 切换时出现不合理幅值跳变；
3. `step_l1` 在尺度切换点是否远高于邻近时间步；
4. PSNR、HF ratio 是否总体随退化逐渐下降；
5. SAM 是否保持合理，不出现由 lift 人为造成的突变。

输出保存在：

```text
outputs/degradation_trajectory/
```

## HSI-MSI Fusion 数据

正式融合实验使用现有真实 SRF：

```bash
python main.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --msi_mode srf
```

仓库已经包含 PaviaU、Houston13、Chikusei 的波长文件和 WorldView-2 SRF CSV；原始 HSI `.mat` 不提交到仓库。

## 当前阶段

UFGNet 网络、损失、训练入口、自动化单元测试、参数量审计和真实数据单 batch sanity script 均已接入。CPU synthetic CI 已通过；下一步是在真实 HSI 数据上执行 `sanity_check_ufgnet.py`，确认中间量和梯度正常后再启动首轮 4× Gaussian+Bicubic 长训练。Physical 与 diffusion 路线保留作为后续扩展，不混入当前 UFGNet 仿真基线。
