# S2Diff: HSI Super-Resolution Experimental Scaffold

当前仓库用于 HSI-MSI Fusion 超分实验。基础代码提供数据读取、SRF、指标、可切换 LR-HSI 退化和渐进物理退化轨迹；具体扩散网络尚未接入。

## UFGNet reproduction protocol

- scale ratio: 4×
- LR-HSI degradation: Gaussian blur + Bicubic downsampling using the repository defaults
- HR-MSI: current repository SRF
- architecture: reproduce UFGNet QIEM / FGM / CCRM / SpeDOB / SpaDOB / FASA without structural simplification
- unsupervised loss: reconstruction + SAM + integrated dual-domain frequency consistency
- metrics: PSNR / SSIM / ERGAS / SAM / CC / RMSE
- the data pipeline, CCRM residual projection, and loss must reuse the same spatial degradation operator and SRF
