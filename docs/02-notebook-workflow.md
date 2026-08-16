# 参数化 Notebook 工作流

[返回文档索引](README.md) · [查看全部参数](03-parameters.md)

项目根目录的 `scMM_workflow.ipynb` 是推荐的交互式入口。所有需要经常修改的路径、实验参数
和分析开关集中在顶部带 `parameters` 标签的单元，其余单元不需要编辑。

## 支持的输入

| 输入 | 示例 | `INPUT_KIND` |
|---|---|---|
| 单个原始谱 | `data/sample.mzML` | `auto` 或 `raw_file` |
| 原始谱目录 | `data/run-01/` | `auto` 或 `raw_dir` |
| 已处理结果 | `results/sample/`，内部含 `.meta` | `auto` 或 `processed` |

自动识别顺序是：受支持扩展名的文件 → 含 `.meta` 的目录 → 普通目录。原始谱目录只读取其
直接子文件，不递归搜索子目录。

目录模式会把所有原始文件视为同一段连续采集过程，按采集时间排序，并构建统一特征轴。
不同实验条件若不应共享同一时间轴，建议分别处理，再使用 Python API 合并。

## 第一次运行

至少修改：

```python
INPUT_PATH = Path("data/example.mzML")
OUTPUT_ROOT = Path("results")
REF_MZ = 734.5929
```

然后从头执行所有单元。建议第一次保留以下默认开关：

```python
RUN_DEISOTOPE = False
IMPUTE_METHOD = None
NORMALIZATION = "total"
RUN_EMBEDDING = True
RUN_CLUSTERING = False
RUN_TRAJECTORY = True
RUN_METABOLITE_TRENDS = True
```

先确认细胞数、特征数、零值比例和 UMAP 合理，再开启去同位素、填补、聚类或注释。

## 执行阶段

### 1. 参数检查

Notebook 检查输入是否存在、类型是否合法、参考 m/z 是否为正数，并创建输出目录。错误会在
读取大文件前暴露。

### 2. 原始谱处理或已有结果载入

原始谱工作流包括：

1. 把每次扫描插值到符合 Orbitrap 分辨率的公共 m/z 网格。
2. 累加谱并进行局部基线/噪声估计。
3. 从去噪总谱提取公共特征峰。
4. 将所有扫描帧对齐到公共峰轴。
5. 在 `REF_MZ` 通道上识别连续的细胞事件。
6. 为每个细胞事件提取特征峰强度，并删除过度稀疏的特征。

若启用 `EXPORT_SUMMED_SPECTRUM`，原始目录还会输出 `total_sum_spec.mzML`，用于检查公共峰。

### 3. 数据变换

Notebook 固定按以下顺序执行：

```text
去同位素 → 异常值过滤 → 缺失值填补 → 归一化
```

先去同位素是因为其判定依赖峰间原始强度关系；归一化放在最后，以免前面的操作破坏每个
细胞的归一化尺度。对于已处理输入，Notebook 不知道历史上做过哪些变换，使用者应避免重复
执行非幂等操作。总量归一化在相同 `scale` 下基本幂等，但填补、去同位素和异常值过滤不是。

### 4. SDF 候选注释

设置 `SDF_PATH` 后，根据精确质量、离子模式和加合物搜索候选。完整候选保存为
`annotation_candidates.csv`，ppm 误差最小的候选写入 `feature_meta`。

这是精确质量候选注释，不是结构确证；同分异构体、同量异位化合物和加合物歧义需要通过
标准品、保留时间或 MS/MS 进一步确认。

### 5. 保存、降维与时间分析

数据先通过 `CyESIData.save()` 保存，再转换为 AnnData。PCA/UMAP、聚类、轨迹和趋势分析产生
的中间结果主要保存在当前内存中的 `engine.adata`，图写入 `FIGURE_DIR`。如需长期保存完整
AnnData 分析状态，可在 notebook 末尾自行执行：

```python
engine.adata.write_h5ad(output_root / "analysis.h5ad")
```

该文件可能较大，因此示例默认不自动生成。

## 三种典型配置

### 快速质控

```python
RUN_DEISOTOPE = False
NORMALIZATION = "total"
RUN_EMBEDDING = True
RUN_CLUSTERING = False
RUN_TRAJECTORY = False
RUN_METABOLITE_TRENDS = False
RUN_STABILITY_QC = True
```

### 完整时间分析

```python
RUN_DEISOTOPE = True
RUN_EMBEDDING = True
RUN_CLUSTERING = True
RUN_TRAJECTORY = True
RUN_METABOLIC_VELOCITY = True
RUN_METABOLITE_TRENDS = True
RUN_TREND_CLUSTERING = True
```

### 只读取已有结果重新画图

```python
INPUT_PATH = Path("results/sample")
INPUT_KIND = "processed"
RUN_DEISOTOPE = False
RUN_OUTLIER_REMOVAL = False
IMPUTE_METHOD = None
NORMALIZATION = None  # 已经归一化时避免重复改变数据
```

## 重复运行与覆盖

`data.save(OUTPUT_ROOT)` 会在 `OUTPUT_ROOT` 下再创建以数据集命名的目录。首次运行使用
`OVERWRITE=False` 可以防止意外覆盖。确认要用新参数替换同名结果时才设置：

```python
OVERWRITE = True
```

这会覆盖 scMM 管理的标准结果文件，但不会自动清空结果目录中的其他文件。

## 输出导航

- 数据矩阵和元数据：[数据模型与输出文件](06-data-and-output.md)
- 图形与分析对象：[降维、聚类与时间分析](05-analysis.md)
- 参数选择：[参数参考](03-parameters.md)
- 运行失败：[常见问题](07-troubleshooting.md)
