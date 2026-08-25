# S2Diff: HSI Super-Resolution Experimental Scaffold

当前仓库用于 HSI-MSI Fusion 超分实验。基础代码提供数据读取、SRF、指标、可切换 LR-HSI 退化和渐进物理退化轨迹；具体扩散网络尚未接入。

## 当前代码

| 文件/目录 | 说明 |
|---|---|
| `data_loader.py` | HSI 读取、patch 构建、LR-HSI 退化、HR-MSI 构建、DataLoader |
| `degradations/` | Bicubic、Gaussian+Bicubic、Physical 退化及 progressive trajectory |
| `check_degradation_trajectory.py` | 在接网络前检查各时间步退化状态 |
| `losses.py` | 当前已实现 SAMLoss；其他一致性损失将在模型阶段补充 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、离散积分权重、HSI→MSI |
| `config.py` | 数据、退化和训练通用配置 |
| `main.py` | 检查数据与退化配置是否正确串联 |

## UFGNet reproduction protocol

当前复现分支采用用户指定实验协议，而不是论文原始 8× 数据退化设置：

- 空间倍率固定为 `4×`；
- 首阶段主实验使用 `gaussian_bicubic`：Gaussian blur + Bicubic downsampling；
- Gaussian 核参数沿用仓库默认设置；
- HR-MSI 使用仓库现有 SRF 构建；
- 网络结构严格按照 UFGNet 论文实现，不简化 QIEM、FGM、CCRM、SpeDOB、SpaDOB、FASA；
- 无监督训练损失严格保留 reconstruction、SAM 与 integrated dual-domain frequency consistency 三项；
- 评价指标固定为 PSNR / SSIM / ERGAS / SAM / CC / RMSE。

为保持无监督观测闭合，数据生成、CCRM residual back-projection 与训练 loss 共用同一个空间退化算子和同一 SRF。
