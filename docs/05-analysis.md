# 降维、聚类与时间分析

[返回文档索引](README.md) · [参数参考](03-parameters.md)

下游分析的高层入口是 `scMM.plot.engine.PlotEngine`。它内部使用 AnnData 保存矩阵、细胞元数据、
特征元数据、低维坐标和分析结果。

公开入口仍集中在 `PlotEngine`，实现则按降维、轨迹、细胞聚类和特征网络四个领域拆分。各领域
共享同一份 `adata` 和图目录，因此方法链及结果键保持一致；维护单个算法时无需理解整个绘图类。

## 初始化

推荐从 `CyESIData` 转换：

```python
from scMM.plot.engine import PlotEngine

adata = data.to_anndata()
engine = PlotEngine.from_adata(adata, fig_path_dir="results/figures")
```

也可以从 DataFrame 初始化，并分别传入 `obs` 和 `var`：

```python
engine = PlotEngine(df, "results/figures", obs=cell_meta, var=feature_meta)
```

图目录会自动创建。大多数绘图方法把 SVG 写入该目录并返回 `self`；数值结果保存在
`engine.adata.obsm` 或 `engine.adata.uns`。

## PCA

```python
X_pca = engine.pca(
    n_components=50,
    scale=True,
    zero_center=True,
    random_state=42,
)
```

| 参数 | API 默认值 | 含义 |
|---|---:|---|
| `n_components` | `50` | 主成分数；自动截断到样本数和特征数 |
| `scale` | `True` | 是否除以特征标准差 |
| `zero_center` | `True` | 是否减去特征均值 |
| `random_state` | `42` | 随机种子 |
| `store_key` | `X_pca` | `adata.obsm` 保存键 |
| `return_model` | `False` | 是否同时返回 sklearn PCA 模型 |

解释方差比例和处理参数保存在 `adata.uns["X_pca_params"]`。如果输入已按特征 z-score，仍可
考虑 `scale=False`，避免重复标准化。

## UMAP

```python
X_umap = engine.umap(
    use_pca=True,
    n_neighbors=15,
    min_dist=0.7,
    metric="euclidean",
    random_state=42,
)
```

| 参数 | API 默认值 | Notebook 值 | 含义 |
|---|---:|---:|---|
| `n_components` | `2` | `2` | 输出维数 |
| `n_neighbors` | `30` | `15` | 局部近邻规模；必须在 2 到细胞数减 1 之间 |
| `min_dist` | `0.3` | `0.7` | 低维空间点的最小距离 |
| `metric` | `euclidean` | `euclidean` | 距离度量 |
| `random_state` | `42` | `42` | 随机种子 |
| `use_pca` | `False` | `True` | 是否使用已有/临时 PCA 作为输入 |
| `pca_key` | `X_pca` | `X_pca` | 已有 PCA 的 `obsm` 键 |
| `pca_n_components` | `30` | 不使用 | 没有已有 PCA 时临时 PCA 维数 |

UMAP 是可视化和近邻结构模型，不应仅凭图上距离断言精确生物学时间或连续动力学。

## 细胞聚类

```python
engine.cluster_cells(
    method="leiden",
    key_added="clusters",
    n_neighbors=15,
    resolution=1.0,
    random_state=0,
)
```

聚类使用 `X_pca` 构建近邻图，并在 `X_umap` 上绘制结果，所以两者必须先存在。

- Leiden 需要 `python-igraph` 和 `leidenalg`。
- Louvain 需要 `networkx`。
- 标签写入 `adata.obs[key_added]`。
- `resolution` 越高通常产生更多小簇，但关系不是固定线性的。

建议报告 PCA 维数、近邻数、算法、分辨率和随机种子，并检查不同参数下结论是否稳定。

## 实验时间与伪时间

原始加载自动创建的 `obs["time"]` 表示采集进程：

- 单文件：`rt / max(rt)`。
- 多文件：按文件采集时间与帧内保留时间映射到整个批次的 0–1 区间。

它不是小时或分钟。若已知真实时间，可添加：

```python
data.peak_meta["true_time_h"] = data.peak_meta["time"] * 12
adata = data.to_anndata()
```

这只有在“整个采集跨度确实对应 12 小时”时才成立。离散实验组时间不应简单假定细胞在组内
均匀变化。

## Palantir 伪时间

```python
engine.run_palantir(
    start_idx=0,
    plotting=True,
    use_obsm="X_umap",
    knn=30,
    num_waypoints=500,
)
```

API 默认值：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `start_idx` | 必填 | 起始细胞的整数位置 |
| `terminal_states` | `None` | 可选终末细胞列表 |
| `n_diff_components` | `10` | 扩散成分数 |
| `knn` | `30` | Palantir 近邻数 |
| `num_waypoints` | `500` | 路标点数量 |
| `scale_components` | `True` | 是否缩放扩散成分 |
| `max_iterations` | `25` | 最大迭代次数 |
| `seed` | `42` | 随机种子 |

起始细胞应由实验先验、时间或标志物确定，而不是为了得到期望轨迹而挑选。结果默认写入
`obs["palantir_pseudotime"]` 和 `obsm["palantir_branch_probs"]`。

## 滑动窗口轨迹

```python
engine.compute_trajectory(
    window_size=100,
    step_size=50,
    cell_dist_key="X_umap",
    parameterization_key="time",
    branch_prob_key=None,
    min_cells_per_window=5,
    plotting=True,
)
```

算法先按参数化列排序细胞，再在每个细胞窗口内计算低维坐标中心。若给定分支概率，会分别对
每个分支进行加权。

| 参数 | API 默认值 | 说明 |
|---|---:|---|
| `window_size` | `100` | 每个窗口的细胞数 |
| `step_size` | `50` | 窗口起点步长，单位也是细胞数 |
| `cell_dist_key` | `X_umap` | 轨迹所在低维坐标 |
| `parameterization_key` | `palantir_pseudotime` | 排序细胞的 `obs` 列 |
| `branch_prob_key` | `palantir_branch_probs` | 分支权重；不存在或设为 `None` 时按单分支处理 |
| `store_key` | `trajectory` | 结果在 `adata.uns` 中的键 |
| `min_cells_per_window` | `5` | 有效窗口/分支所需最少细胞 |

Notebook 使用实验 `time` 而非 Palantir 伪时间，并采用更大的 1000/300 窗口作为高细胞数
CyESI 数据的起点。

## 代谢速度

```python
engine.metabolic_velocity(
    window_size=100,
    step_size=50,
    parameterization_key="time",
    plot=True,
)
```

每个时间窗口内，对每个特征做强度随时间的一阶局部回归。所有特征斜率组成速度向量，其
欧氏范数作为总体代谢速度。结果在 `adata.uns["metabolic_velocity"]`：

- `state_centers`
- `velocity_field`
- `speeds`
- `time_centers`
- `counts`
- `window_starts`

如果窗口内所有时间相同，回归分母为零，该窗口无法产生有效速度。

## 代谢趋势

```python
engine.plot_metabolite_trends(
    parameterization_key="time",
    window_size=100,
    step_size=50,
    kernel_stat="median",
    feature_name_key="mz",
    plot_top_n=30,
)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `kernel_stat` | `median` | 窗口聚合方法：`median`、`mean`、`sum` |
| `feature_name_key` | `None` | 用作图中名称的 `var` 列；`None` 使用 var names |
| `plot_top_n` | `None` | 是否绘制排名靠前的 N 个特征 |
| `cmap` | `viridis` | 热图颜色映射 |

每个特征的池化趋势、时间中心、相关系数、p 值、q 值和排名保存在
`adata.uns["metabolite_trends"]`。热图为了可视化会对每一行做 z-score，因此颜色表示相对趋势，
不是原始强度。

## 趋势聚类

```python
engine.plot_trend_clusters(
    metric="correlation",
    cluster_method="leiden",
    top_k=150,
    resolution=0.5,
)
```

支持的底层聚类方法包括 Leiden、KMeans 和层次聚类，具体额外参数通过 `**kwargs` 传入。
相关距离适合比较曲线形状；欧氏距离同时受幅度影响。无法形成有限趋势的特征会标记为无效，
不应强制归入某个生物学簇。

## 特征网络

```python
embedding = engine.feature_network(
    name_key="annotation_name",
    class_key="lipid_class",
    metric="pearson",
    n_neighbors=15,
)
```

该方法把“特征”而不是“细胞”嵌入二维空间：

- `pearson`：根据绝对相关性构造距离。
- `euclidean`：直接比较跨细胞强度向量。
- 至少需要 3 个有效命名特征。
- `n_neighbors` 必须在 2 到特征数减 1 之间。

网络图是探索性可视化，不代表已知生化反应边。

## 轻量降维接口

`scMM.plot.embedding.dimension_reduction()` 提供 PCA、UMAP、t-SNE、Isomap、LLE 的统一绘图接口，
并支持按来源标签、连续值或 DBSCAN/自定义模型着色。需要完整轨迹和结果持久化时优先使用
`PlotEngine`。

```python
from scMM.plot.embedding import dimension_reduction

result = dimension_reduction(
    data,
    method="umap",
    color="categorical",
    reduce_kwargs={"n_neighbors": 15, "min_dist": 0.1},
    plot_kwargs={"s": 8, "alpha": 0.8},
)
embedding = result["X_emb"]
ax = result["ax"]
```

各方法未覆盖时的内置起点：

| 方法 | 关键默认值 |
|---|---|
| PCA | `n_components=2`, `svd_solver="auto"` |
| UMAP | `n_neighbors=15`, `min_dist=0.1`, `n_components=2`, `metric="euclidean"`, `random_state=42` |
| t-SNE | `n_components=2`, `perplexity=30`, `learning_rate="auto"`, `init="pca"`, `random_state=42` |
| Isomap | `n_components=2`, `n_neighbors=15`, `metric="euclidean"` |
| LLE | `n_components=2`, `n_neighbors=15`, `method="standard"` |

`color="categorical"` 使用数据来源标签；`color="cluster"` 默认使用 DBSCAN，也可在
`cluster_kwargs["method"]` 中传入兼容的聚类器；一维 NumPy 数组用于连续值着色。接口会复制
聚类参数后再取出 `method`，因此可以安全地复用调用方的参数字典；聚类器必须为每个观测返回一个
标签。

## 原始谱与 EIC 绘图

`scMM.plot.msplot` 提供不依赖 `PlotEngine` 的质谱质控图。

### 提取离子流

```python
import matplotlib.pyplot as plt
from scMM.plot.msplot import eic

fig, ax = plt.subplots()
ax, (x, intensity) = eic(
    ax,
    aligned_frames,
    mz=734.5929,
    ppm_tol=5.0,
    time=frame_times,  # 可省略，省略后横轴为行索引
)
```

`eic` 会汇总容差内的所有列；找不到 m/z 时抛出错误，而不是静默选择最近峰。

### 绘制矩阵汇总谱

```python
from scMM.plot.msplot import plot_ms

fig, ax = plt.subplots()
plot_ms(ax, aligned_frames, frame_range=(0, 1000))
```

`frame_range=None` 默认使用全部行，范围采用 Python 的左闭右开语义。

### 绘制单个 PyOpenMS 谱

```python
from scMM.plot.msplot import plot_spectrum

fig, ax = plot_spectrum(
    spectrum,
    mz_range=(100, 1000),
    top_n_labels=10,
    normalize=True,
    exclusion_window=10.0,
    linewidth=1.0,
    save_path="results/figures/spectrum.svg",
)
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `top_n_labels` | `0` | 标注最强的 N 个相互分离峰；0 不标注 |
| `mz_range` | `None` | 显示质量范围 |
| `intensity_range` | `None` | 手动 y 轴范围 |
| `normalize` | `False` | 是否除以显示范围内最大强度 |
| `exclusion_window` | `10.0` | 被标注峰之间的最小 m/z 间隔 |
| `label_fmt` | `{:.4f}` | m/z 标签格式 |
| `figsize` | `(10, 4)` | 未传入轴时创建的图尺寸 |
| `linewidth` | `1.0` | 谱线宽度 |

绘图前会移除非有限峰并按 m/z 排序；归一化只使用 `mz_range` 内最终显示的峰。峰标签按强度选择，
同时遵守 `exclusion_window`，避免在相邻峰上堆叠标注。

### 全局绘图样式

导入 `scMM.plot` 不会修改 Matplotlib 全局配置。显式调用：

```python
from scMM.plot import configure_plotting

configure_plotting()  # 使用 scMM 字号默认值
configure_plotting("fonts/MyFont.ttf")  # 可选自定义字体
```

字体路径必须指向存在的文件。为了结果可重复，正式图形应记录字体文件或使用环境中稳定可用的字体。

下一步：[数据模型与输出](06-data-and-output.md)。
