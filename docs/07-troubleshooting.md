# 常见问题与参数调优

[返回文档索引](README.md) · [参数参考](03-parameters.md)

## 推荐的调参顺序

一次只调整一类参数，并记录细胞数、特征数、零值比例和代表性图形：

1. 确认 mzML/mzXML 可读、MS level 和质量范围正确。
2. 确认 `REF_MZ` 附近确实有稳定峰。
3. 调整 `CELL_SNR`，先得到可信的细胞事件。
4. 调整 `MS_PEAK_SNR`、`PEAK_SNR` 和 `MAX_ZERO_FRAC`，控制特征质量。
5. 检查 ppm 误差后调整 `PPM_TOL`。
6. 再评估去同位素、异常值、填补和归一化。
7. 最后调整 UMAP、聚类和滑动窗口参数。

如果同时降低所有 SNR、放宽 ppm 并提高零值容忍度，虽然会得到更多特征，但无法判断新增信号
来自哪个选择，也更容易把噪声当作生物学变化。

## 输入与安装

### `ModuleNotFoundError: pyopenms` 或科学计算包缺失

确认当前解释器属于项目环境：

```bash
uv sync --locked --all-extras --dev
uv run --locked python -c "import sys; print(sys.executable)"
uv run --locked python -c "import pyopenms, numpy, pandas, anndata"
```

解释器应位于仓库的 `.venv`。Notebook 中还应检查右上角内核是否为 `scMM`/`scmm`，或直接
选择 `.venv/bin/python`；终端中的环境不会自动切换已经打开的 Notebook 内核。

### 输入目录没有发现 mzML/mzXML

- 只支持 `.mzML`、`.mzXML`，扩展名大小写均可。
- `load_from_filelist()` 和 `batch_process()` 只扫描目录直接子文件，不递归。
- 检查 `INPUT_PATH` 是否误指向上一级目录。
- 已处理目录应包含 `.meta`；否则 `INPUT_KIND="auto"` 会把它当作原始目录。

### `No spectra found`

常见原因是文件为空、目标 MS level 不存在，或所有扫描都落在底层汇总函数默认 m/z 范围之外。
先用 PyOpenMS 检查谱数和 MS level；需要非默认质量范围时使用底层 `sum_spec(mz_range=...)`。

## 细胞事件识别

### 没有识别到细胞

按顺序检查：

1. `REF_MZ` 是否写错、极性是否匹配。
2. 最近的实际特征离参考 m/z 有多远。
3. 参考离子的 EIC 是否存在细胞状尖峰。
4. `CELL_SNR` 是否过高。
5. `BASELINE_FILTER_SIZE` 是否过小，以至于基线跟随了细胞峰。
6. 合并谱 `MS_PEAK_SNR` 是否先把参考峰删除。

可以从 `CELL_SNR=3`、`MS_PEAK_SNR=5` 做诊断，但正式参数应结合空白假阳性恢复到更保守的值。

### 识别到过多细胞或长连续区间

- 增大 `CELL_SNR`。
- 检查参考离子是否在背景/溶剂中持续存在。
- 减小过宽的基线窗口，确认局部漂移得到合理估计。
- 检查离子源是否存在持续喷雾团块或饱和。
- 使用稳定性质控图观察事件是否集中在异常采集阶段。

### 细胞数合理，但特征太少

- 降低 `PEAK_SNR`，例如从 3 降到 2。
- 降低 `MS_PEAK_SNR`，避免公共特征轴过早丢失弱峰。
- 提高 `MAX_ZERO_FRAC`，允许更稀有的特征。
- 检查 `PPM_TOL` 是否小于实际质量误差。

每次只修改一个参数，并使用空白或随机区间估计新增特征中的噪声比例。

### 特征过多、矩阵极度稀疏

- 增大 `MS_PEAK_SNR` 或给 `prominence_ratio` 设置小的正值。
- 增大 `PEAK_SNR`。
- 降低 `MAX_ZERO_FRAC`，例如从 0.90 调到 0.80。
- 检查分辨率是否设置得远高于真实仪器分辨率。
- 检查 ppm 容差过大是否把噪声错误投影到候选峰。

## 保存与输出

### `FileExistsError`

`save("results")` 会写入 `results/<数据集名>`。同名目录存在时默认中止，以保护结果。

- 想保留旧结果：更换 `OUTPUT_ROOT`。
- 明确替换标准结果文件：设置 `OVERWRITE=True` 或 `overwrite=True`。
- 不要把输出根目录误写成现有数据集目录，否则可能产生重复嵌套。

### 找不到刚保存的结果

使用 `save()` 的返回值：

```python
result_dir = data.save("results")
print(result_dir)
```

实际路径比传入路径多一级数据集名称。

### pickle 与 CSV 重新载入结果不一致

scMM 优先读取 pickle，因为它保留数值列名、索引和可空类型。CSV 是交换格式；经过其他软件写回
后，列类型和空值可能变化。不要在仍保留旧 pickle 时只修改 CSV 并期待加载修改后的 CSV。

## 去同位素与注释

### 去同位素删除了可疑的真实峰

先使用 `inplace=False` 审计：

```python
audit = data.deisotope(inplace=False)
display(audit["final_table"])
```

然后考虑：

- 提高 `r_square_threshold`。
- 收紧 `ppm_tol`。
- 保持 `merge_mode="keep_parent"`。
- 设置 `remove=False` 只标注关系。
- 稳定同位素示踪实验直接关闭该步骤，或建立实验专用规则。

共调控代谢物可能同时具有很高相关性和近似质量差，因此统计关系不能替代结构证据。

### SDF 注释没有命中

- 确认 `ION_MODE` 与采集极性一致。
- 确认数据库记录含可解析的 `EXACT_MASS`。
- 检查需要的加合物是否在默认列表中。
- 用标准品估计实际质量误差，再决定是否放宽 `ANNOTATION_PPM`。
- 检查观测的是多电荷离子、碎片还是数据库未覆盖的化合物。

### 一个 m/z 有很多候选

这是精确质量注释的正常情况。不要只保留名称而丢弃：

- 加合物
- 理论 m/z
- ppm 误差
- 分子式
- 数据库 ID

使用 MS/MS、标准品和保留时间缩小候选范围。

## 降维与聚类

### UMAP 报近邻数错误

`n_neighbors` 必须至少为 2 且小于细胞数。Notebook 会自动截断；直接调用 API 时需手动设置：

```python
n_neighbors = min(15, adata.n_obs - 1)
```

少于 3 个细胞不适合运行 UMAP。

### 缺少 Leiden/Louvain 依赖

```bash
uv sync --locked --all-extras --dev
```

然后重启 Notebook 内核。项目声明的是 `python-igraph`；不要手工安装另一个同名 `igraph` 包。

### UMAP 每次变化

- 固定 `random_state`。
- 固定输入矩阵、归一化顺序、PCA 维数和软件版本。
- 避免在 UMAP 前以不固定随机种子的异常值算法修改细胞集合。

即使固定随机种子，不同 UMAP/Numba 版本也可能产生数值差异，所以应归档环境定义。

### 聚类主要反映文件来源

这可能是实验条件，也可能是批次效应。检查：

- 按 `label` 给 UMAP 着色。
- 比较每个文件的总强度、细胞数和检出率。
- 确认所有文件使用一致峰轴和处理参数。
- 在生物学解释前进行适合设计的批次评估/校正。

scMM 的 `alignwith()` 只做 m/z 对齐，不做批次校正。

## 时间轨迹与趋势

### `time` 到底是什么

它通常是 0–1 的采集进程，不是实际小时。多文件情况下依赖文件采集时间戳；时间戳缺失时使用
文件修改时间，这可能受文件复制影响。正式时间分析应核对 `.meta` 中的时间来源。

### 轨迹或趋势只有一个窗口

当 `WINDOW_SIZE >= 细胞数` 时只产生一个全局窗口。降低窗口大小和步长，并确保每个窗口仍有
足够细胞。

### 代谢速度出现 NaN

- 窗口内参数化时间可能完全相同。
- 输入矩阵或时间列可能含 NaN/Inf。
- 窗口可能只有一个细胞。
- 离散组别编码不适合直接进行局部连续回归。

### 趋势热图颜色和原始强度不一致

热图对每个特征沿时间做行内 z-score，显示相对升降趋势。原始池化强度位于：

```python
engine.adata.uns["metabolite_trends"]["pooled"]
```

## 性能和内存

内存压力通常来自“扫描数 × 公共 m/z 网格”和“细胞数 × 特征数”。处理建议：

- 先用单文件验证，不要直接并行整个批次。
- 把 `N_JOBS` 从 `-1` 降到 1–4。
- 使用与仪器一致的 `RESOLUTION_200`，不要人为设得过高。
- 适当降低 `RESAMPLE_POINTS_PER_FWHM`，但应检查峰形和对齐。
- 使用更严格的合并谱 SNR 和稀疏特征过滤。
- 分批独立处理后用 `concat()` 合并，便于定位资源瓶颈。

如果问题仍无法定位，使用 `LOG_LEVEL="DEBUG"` 或 CLI `--verbose`，并记录输入文件数量、扫描数、
细胞数、特征数和出错阶段。
