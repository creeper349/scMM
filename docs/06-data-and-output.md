# 数据模型与输出文件

[返回文档索引](README.md) · [Python API](04-python-api.md)

## `CyESIData` 数据模型

一个处理后的数据集由四部分组成。

### `data`

pandas DataFrame，行是细胞事件，列是 m/z 特征，值是对应强度。

- 原始处理后未检出的峰通常记为 0。
- 列名可转换为浮点 m/z。
- 去同位素、填补和归一化会修改该矩阵。

### `peak_meta`

与 `data` 行一一对应的细胞元数据。常见列：

| 列 | 含义 |
|---|---|
| `rt` | 原始文件内的保留/扫描时间 |
| `time` | 归一化到 0–1 的采集进程 |
| `label` | 来源文件名去掉扩展名后的标签 |

具体列还取决于 PyOpenMS 读取和帧对齐结果。删除异常细胞时 `peak_meta` 会同步过滤。

### `feature_meta`

与 `data` 列一一对应的特征元数据。基础列为：

| 列 | 含义 |
|---|---|
| `mz` | 特征的数值 m/z |

运行去同位素后还可能包括：

- `deisotope_role`：`unique`、`parent` 或 `isotope`。
- `isotope_parent`、`isotope_order`。
- `isotope_slope_A`、`isotope_r_square`、`isotope_ppm_error`。
- `isotope_children` 和每阶 M+n 的详细字段。

参数化 Notebook 注释后还可能包括：

- `annotation_name`
- `annotation_formula`
- `annotation_adduct`
- `annotation_ppm_error`

### `file_meta`

普通字典，保存数据集级元数据：

- `name`
- `ref_mz`
- `length`
- 仪器和采集时间，或多文件的 `per_file_meta`
- 去同位素处理参数和删除数量

它不能自动记录 Notebook 中的所有分析开关。正式分析建议额外保存参数单元、运行日志和 Git
提交 ID。

## 保存目录

```python
result_path = data.save("results")
```

假设 `data.get_name()` 为 `sample`，会产生：

```text
results/
└── sample/
    ├── .meta
    ├── data.pkl
    ├── data.csv
    ├── peak_meta.pkl
    ├── peak_meta.csv
    ├── feature_meta.pkl
    └── feature_meta.csv
```

| 文件 | 用途 |
|---|---|
| `.meta` | JSON 数据集元数据；也是 scMM 已处理目录的识别标志 |
| `data.pkl` | 高保真、快速重新加载的强度矩阵 |
| `data.csv` | 便于人工检查和跨软件交换的强度矩阵 |
| `peak_meta.pkl/.csv` | 细胞元数据 |
| `feature_meta.pkl/.csv` | 特征元数据 |

读取时优先选择 pickle；CSV 主要用于交换。CSV 往返可能把数值列名或可空数据类型转换成字符串。

## Notebook 输出

默认目录关系：

```text
results/
├── <数据集名>/
│   └── scMM 标准数据文件
├── figures/
│   ├── umap.svg
│   ├── trajectory.svg
│   ├── metabolic_velocity_speed.svg
│   ├── metabolite_trends_top<N>.svg
│   ├── leiden_<key>_umap.svg
│   ├── trend_clusters_<method>_<metric>.svg
│   ├── stability_monitor_boxplot.svg
│   ├── stability_cell_counts.svg
│   ├── stability_median_correlation.svg
│   └── feature_ratio_umap.svg
├── annotation_candidates.csv
└── total_sum_spec.mzML
```

只有启用相应功能时才会生成对应文件。`total_sum_spec.mzML` 仅适用于原始目录输入。

## AnnData 映射

`data.to_anndata()` 产生：

```text
AnnData
├── X      ← 细胞 × 特征强度
├── obs    ← peak_meta + source_index
├── var    ← feature_meta
├── raw    ← 转换时的 X/obs/var 快照
├── obsm
│   ├── X_pca
│   ├── X_umap
│   └── palantir_branch_probs
└── uns
    ├── X_pca_params
    ├── X_umap_params
    ├── trajectory
    ├── trajectory_metadata
    ├── metabolic_velocity
    └── metabolite_trends
```

`adata.raw` 是调用 `to_anndata()` 时的数据快照，不一定是原始仪器强度：如果先对 `CyESIData`
归一化或去同位素，raw 同样反映变换后的数据。

## 输出可重复性清单

每次正式运行建议同时归档：

1. 原始 mzML/mzXML 的只读副本或校验和。
2. `environment.yml` 和 `pyproject.toml`。
3. Git 提交 ID：`git rev-parse HEAD`。
4. Notebook 参数单元或单独的参数 YAML/JSON。
5. 日志输出。
6. scMM 标准结果目录。
7. 完整候选注释，而不只保留最佳候选。
8. 若后续依赖 `engine.adata` 中的分析结果，额外保存 H5AD。

## 建议的项目数据目录

原始数据通常较大，不建议直接提交 Git：

```text
project/
├── raw-data/           # 只读原始谱，不提交 Git
├── metadata/           # 样本表、实验设计、SDF 来源说明
├── results/
│   ├── processed/
│   ├── figures/
│   └── tables/
├── notebooks/          # 如有实验专用 notebook
└── run-parameters/     # 每次分析的参数快照
```

本仓库根目录的 `scMM_workflow.ipynb` 是通用模板；实验专用副本可以放到分析项目中，避免把
大量执行输出和本地路径写回软件仓库。

下一步：[常见问题与调优](07-troubleshooting.md)。
