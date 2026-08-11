# Python API 与批处理

[返回文档索引](README.md) · [参数参考](03-parameters.md)

## 单个原始文件

```python
from scMM.file.data import CyESIData

data = CyESIData.load_from_file(
    "data/sample.mzML",
    ref_mz=734.5929,
    ppm_tol=10,
    resolution=35_000,
    resample_points_per_fwhm=5,
    ms_peak_snr_threshold=10,
    cell_snr=5,
    peak_snr=3,
    baseline_filter_size=50,
    max_zero_frac=0.9,
)
```

`load_from_file()` 完成谱读取、总谱构建、去噪、公共峰提取、扫描对齐和细胞事件识别。
返回的对象可以通过 `len(data)` 查看细胞数，通过 `data.data.shape` 查看完整矩阵形状。

## 多个原始文件作为一个连续数据集

```python
data = CyESIData.load_from_filelist(
    "data/run-01",
    ref_mz=734.5929,
    ppm_tol=10,
    n_jobs=4,
    cell_snr=5,
    peak_snr=3,
)
```

该方法先从目录中所有直接子文件构建一个合并谱和公共特征轴，然后并行对齐每个文件。文件按
采集时间排序，所有细胞共享 0–1 的全局 `time`。它适合属于同一连续实验的多段采集。

## 重新载入已有结果

```python
data = CyESIData.load_from_processed("results/sample")
# 等价的兼容写法：data = CyESIData("results/sample")
```

目录中必须有 `.meta`，并至少有 `data.pkl` 或 `data.csv`。读取时优先使用 pickle；pickle 能更
可靠地保留列类型和 pandas 数据类型。

## 数据变换方法链

除 `deisotope(inplace=False)` 外，以下方法都会修改当前对象并返回自身，因此可以串联：

```python
data = (
    data.deisotope(ppm_tol=1.0, r_square_threshold=0.95)
    .remove_outlier(contamination=0.01, random_state=42)
    .impute(method="knn", missing_values=0, n_neighbors=5)
    .normalize(method="total", scale=1.0)
)
```

建议逐步运行并记录每一步前后的细胞数、特征数和零值比例，而不是第一次就全部串联。

### 检查去同位素结果而不修改数据

```python
audit = data.deisotope(
    ppm_tol=1.0,
    r_square_threshold=0.95,
    inplace=False,
)

candidate_pairs = audit["candidate_table"]
accepted_pairs = audit["final_table"]
```

`final_table` 包含母峰、子峰、同位素阶数、ppm 误差、回归斜率、R² 和允许的最大强度比。
`inplace=False` 不会写入 `data`、`feature_meta`、`file_meta` 或 `deisotope_result`，因此适合先做参数
审计。`merge_mode="sum"` 会把确认的子峰强度累加到母峰；是否从结果中删除子峰仍由 `remove`
独立控制。

## 质量注释

```python
hits = data.get_annotation(
    "data/structures.sdf",
    ppm_tol=5,
    search_mode="pos",
    max_results_per_mz=5,
)
hits.to_csv("results/annotation_candidates.csv", index=False)
```

`get_annotation()` 只返回候选表，不自动修改 `feature_meta`。参数化 Notebook 会额外把每个 m/z
误差最小的候选映射到特征元数据。
批量 `search()` 即使没有命中也会返回带稳定候选列的空 DataFrame，便于直接拼接或写出；候选
生成、ppm 排序和每个查询的数量限制对自定义单/多电荷加合物采用相同流程。

若只需查询少量质量，可直接使用：

```python
from scMM.util.annotation import SDFMzSearcher

searcher = SDFMzSearcher("data/structures.sdf")
hits = searcher.search([734.5929, 760.5839], ppm_tol=5, mode="pos")
```

## 访问数据

```python
print(data.get_name())
print(data.data.shape)
print(data.peak_meta.head())
print(data.feature_meta.head())

# 返回最接近目标 m/z 的整列强度；这是最近邻选择，不施加 ppm 上限
intensity = data[734.5929]

# 获取来源标签，并可选映射成实验组名
labels = data.get_labels({"sample_01": "control", "sample_02": "treated"})
```

需要严格质量容差时，不要直接依赖 `data[mz]`；应先计算最近特征的 ppm 误差并检查是否可接受。

## 保存与重载

```python
result_dir = data.save("results", overwrite=False)
print(result_dir)  # 通常为 results/<data.get_name()>

same_data = CyESIData.load_from_processed(result_dir)
```

`save()` 参数是根目录，不是最终目录。再次保存同名数据时默认抛出 `FileExistsError`；只有明确
接受覆盖时使用 `overwrite=True`。

## 转换为 AnnData

```python
adata = data.to_anndata()
```

映射关系：

- `data.data` → `adata.X`
- `data.peak_meta` → `adata.obs`
- `data.feature_meta` → `adata.var`
- 原始矩阵快照 → `adata.raw`

细胞 ID 会重建为 `cell_0`、`cell_1` 等稳定唯一名称；原始 DataFrame 索引保存在
`adata.obs["source_index"]`。

## 两种批处理模式的区别

### 模式 A：共享公共峰轴

```python
combined = CyESIData.load_from_filelist("raw-data", ref_mz=734.5929, n_jobs=4)
```

所有文件先共同生成公共峰轴，再形成一个连续数据集。适合连续采集或希望从一开始就共享特征
定义的数据。

### 模式 B：先独立处理，再对齐合并

```python
from scMM.file.batch import batch_process, concat

paths = batch_process(
    "raw-data",
    "per-file-results",
    n_jobs=4,
    ref_mz=734.5929,
    ppm_tol=10,
)

combined = concat(
    "per-file-results",
    "combined-results",
    ppm_tol=5,
    ref_idx=0,
    mz_merge_options="union",
)
```

每个原始文件独立产生特征轴和结果目录，然后 `concat()` 按 ppm 对齐。适合需要检查各文件处理
质量、单独重跑失败文件，或文件并不构成一条连续时间轴的情况。

注意：`batch_process(n_jobs=...)` 的并发发生在文件之间；传给各文件加载函数的其他参数放在
`**kwargs`。不要在同一层重复传入两个 `n_jobs`。

## 手动合并两个数据集

```python
left = CyESIData.load_from_processed("results/control")
right = CyESIData.load_from_processed("results/treated")

left.alignwith(right, ppm_tol=5, mz_merge_options="union")
```

该操作原地修改 `left`：

- `union`：保留两边未匹配特征，不存在的位置填 0。
- `ref`：只保留 `left` 的特征轴。

这只是 m/z 对齐和行拼接，不会自动进行批次校正、重新归一化或时间重标定。

## 命令行批处理

```bash
scmm-process raw-data/ results \
  --ref-mz 734.5929 \
  --ppm-tol 10 \
  --resolution 35000 \
  --cell-snr 5 \
  --peak-snr 3 \
  --jobs 4 \
  --verbose
```

CLI 的目录模式等价于 `load_from_filelist()`，不是 `batch_process()`。它输出一个合并数据集。

## 底层谱工具

高级工作流不能满足时，可以组合 `scMM.file.io`：

- `load_single_file()`：读取 mzML/mzXML 和文件元数据。
- `sum_spec()`：在变分辨率 Orbitrap 网格上汇总谱。
- `sum_spectrum_from_file()`：读取并汇总单文件。
- `extract_peaks()`：从汇总谱提取质心峰。
- `align_frame()`：把每个扫描对齐到指定 m/z 列表。
- `pack_specs()`：把多个 `MSSpectrum` 包装为实验对象。
- `save_spectra()`：保存 mzML/mzXML。

底层 API 不自动完成细胞识别和元数据封装。一般分析优先使用 `CyESIData`。

## 高级矩阵去噪工具

`scMM.util.denoise` 提供两个不属于默认预处理链的研究型工具：

```python
from scMM.util.denoise import peak_recon, r1_decomposition

a, b = r1_decomposition(X, tol=1e-6, max_iter=100)
reconstructed, noise_sigma = peak_recon(
    signal,
    baseline,
    lam=0.5,
    sigma_min=1e-3,
    tau=2.0,
    max_iter=50,
    n_jobs=4,
)
```

- `r1_decomposition` 用两个向量的外积近似输入矩阵，适合检查或提取秩一结构。
- `peak_recon` 在给定信号矩阵 `S` 和同形状基线矩阵 `B` 后，逐特征分离稀疏正峰和噪声。
- `lam` 和 `tau` 越大通常越强调稀疏性；没有通用推荐值，应使用合成峰、空白和标准品验证。
- 两个输入必须是有限、非空且形状一致的二维数组。
- 这些工具不会自动更新 `CyESIData` 的 `peak_meta` 或 `feature_meta`，因此不应把返回矩阵直接替换
  `data.data`，除非已经确认维度和语义保持一致。

下一步：[下游分析](05-analysis.md) 和 [数据输出](06-data-and-output.md)。
