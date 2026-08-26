# Comparison Experiments

`comparison/` 专门用于存放所有对比方法。后续新增对比实验时，每个方法单独建立一个子目录，不再把模型代码、权重和实验结果散放在仓库根目录。

推荐结构：

```text
comparison/
├── README.md
├── EMR-Diff/
│   ├── ... source code ...
│   ├── checkpoints/
│   └── outputs/
└── <OtherMethod>/
    ├── ... source code ...
    ├── checkpoints/
    └── outputs/
```

## 目录规则

1. 每个对比方法使用 `comparison/<Method>/` 作为自己的工作根目录。
2. 训练产生的模型权重统一保存到该方法自己的 `comparison/<Method>/checkpoints/` 下，不保存到仓库根目录的公共 checkpoint 目录。
3. 测试结果、重建结果、指标文件和中间实验输出统一保存到该方法自己的 `comparison/<Method>/outputs/` 下。
4. 不同方法之间只共享仓库根目录的数据、退化算子、SRF、评价指标或其他明确需要统一的实验协议代码；方法自身的模型文件、配置、日志、权重和结果保持隔离。
5. 新增对比方法时优先保持原开源代码结构，在该方法子目录内做适配，避免为每个方法创建额外 Git 分支。
6. `checkpoints/` 和 `outputs/` 默认不提交 Git；需要长期记录的最终指标建议整理成轻量文本、CSV 或 Markdown 后再决定是否提交。

## 当前方法

- `EMR-Diff/`：EMR-Diff 对比实验及 UFGNet 协议适配。
