# Adult Lab

基于 Adult 数据集的 ID3 决策树与随机森林分类实验工程，使用 `uv` 管理依赖与运行。

## 快速开始

```bash
uv sync --extra dev
uv run adult-lab run-all
```

## 常用命令

```bash
uv run adult-lab download-data
uv run adult-lab profile-data
uv run adult-lab train --model tree
uv run adult-lab train --model forest
uv run adult-lab evaluate --model tree
uv run adult-lab evaluate --model forest
uv run adult-lab run-all
```

## 输出目录

- `artifacts/metrics.json`
- `artifacts/predictions.csv`
- `artifacts/model_tree.json`
- `artifacts/model_forest.json`
- `artifacts/figures/*.png`
- `reports/experiment_report.md`
