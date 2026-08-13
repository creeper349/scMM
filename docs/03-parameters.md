# 参数默认值、含义与推荐

[返回文档索引](README.md) · [Notebook 工作流](02-notebook-workflow.md)

本页区分三类默认值：

- **Notebook 默认值**：`scMM_workflow.ipynb` 顶部参数单元中的值。
- **Python API 默认值**：函数签名中的值。
- **CLI 默认值**：`uv run --locked scmm-process --help` 显示的值。

推荐值是起始范围，不是仪器或实验的通用最优值。最终参数应由标准品、空白、批内质控和已知
参考离子验证。

## Notebook：路径与运行控制

| 参数 | 默认值 | 含义 | 推荐与注意事项 |
|---|---:|---|---|
| `INPUT_PATH` | `data/example.mzML` | 原始文件、原始目录或已处理目录 | 必须修改；建议使用项目根目录下的相对路径或明确的绝对路径 |
| `INPUT_KIND` | `auto` | 输入类型 | 通常保持 `auto`；目录同时含 `.meta` 和原始谱时会识别为 `processed` |
| `OUTPUT_ROOT` | `results` | 数据、候选注释等输出的根目录 | 不要指向原始数据目录 |
| `FIGURE_DIR` | `OUTPUT_ROOT/figures` | 所有 SVG 图目录 | 多组参数比较时建议使用不同子目录 |
| `OVERWRITE` | `False` | 是否允许写入已存在的同名结果目录 | 调参阶段确认旧结果可替换后才设为 `True` |
| `LOG_LEVEL` | `INFO` | Python 日志级别 | 排错时使用 `DEBUG`；常规批处理使用 `INFO` |

## 原始谱与细胞检测

这些参数只在输入为原始 mzML/mzXML 时使用。

| Notebook 参数 | API 参数 | 默认值 | 含义 | 推荐与调优方向 |
|---|---|---:|---|---|
| `REF_MZ` | `ref_mz` | `734.5929`（示例值） | 用于识别细胞事件的参考离子 | **必须按实验修改**；选择稳定、细胞中高响应、空白中低响应的离子 |
| `PPM_TOL` | `ppm_tol` | `10.0` / API `10` | 扫描帧与公共峰轴的质量容差 | 高分辨且已校准数据可从 5 ppm 开始；漂移明显时尝试 10 ppm；过大会混合邻峰 |
| `RESOLUTION_200` | `resolution` | `35000` | 仪器在 m/z 200 处的分辨率 | 使用采集方法标称值；设得过高会增加网格和内存，过低会合并近邻峰 |
| `RESAMPLE_POINTS_PER_FWHM` | `resample_points_per_fwhm` | `5.0` | 每个峰半高宽的插值采样点数 | 4–6 通常是合理起点；增大提高网格密度但增加内存和时间 |
| `MS_PEAK_SNR` | `ms_peak_snr_threshold` | `10.0` | 合并谱保留峰的 SNR 阈值 | 特征过多/噪声多时增大；弱峰大量丢失时减小到 5 左右并检查空白 |
| — | `prominence_ratio` | `None` | `find_peaks` 的相对峰突出度阈值 | `None` 不施加突出度限制；噪声峰过多时可尝试 `0.001`–`0.01` |
| — | `distance` | `3` | 合并谱中相邻候选峰的最小网格点间距 | 通常保持默认；它不是 Da 或 ppm |
| `CELL_SNR` | `cell_snr` | `5.0` | 参考通道信号相对局部基线的细胞判定倍数 | 假阳性多时增大；已知细胞事件漏检时减小，并结合原始 EIC 检查 |
| `PEAK_SNR` | `peak_snr` | `3.0` | 单个细胞事件内特征峰相对基线的阈值 | 2–5 可作为调试范围；越低矩阵越密但噪声越多 |
| `BASELINE_FILTER_SIZE` | `baseline_filter_size` | `50` | 沿扫描帧方向估计基线的中值滤波窗口 | 应明显宽于单个细胞事件；慢漂移可增大，快速基线变化可减小 |
| `MAX_ZERO_FRAC` | `max_zero_frac` | `0.90` | 允许特征为零的最大细胞比例 | `0.90` 表示至少约 10% 细胞检出；探索稀有特征时可升至 0.95–0.99 |
| `N_JOBS` | `n_jobs` | `-1` | 多文件或部分特征处理的并行工作数 | `-1` 使用所有 CPU；共享服务器或内存紧张时设置 1–4 |

### 合并谱底层去噪参数

高层加载接口只直接暴露 `ms_peak_snr_threshold`。需要底层控制时可调用
`scMM.util.peak.filter_spectrum`：

| 参数 | API 默认值 | 含义 | 推荐 |
|---|---:|---|---|
| `baseline_window` | `101` | 局部基线窗口，偶数会自动调整为奇数 | 覆盖数个峰宽，不能小到跟随峰顶 |
| `noise_window` | `101` | 用 MAD 估计局部噪声的窗口 | 通常与基线窗口相近 |
| `baseline_quantile` | `0.1` | 窗口内作为基线的分位数 | 0.05–0.2；峰密集时使用较低值 |
| `snr_threshold` | `3.0` | 保留信号的 SNR | 高层处理会传入 Notebook 的 `MS_PEAK_SNR=10.0` |
| `keep_negative` | `False` | 是否保留扣基线后的负值 | 强度矩阵通常保持 `False` |
| `baseline_stride` | `10` | 间隔多少点计算一次基线锚点 | 增大可提速，过大可能忽略快速基线变化 |

## 数据变换

### 去同位素

Notebook 默认 `RUN_DEISOTOPE=False`；开启后使用 `DEISOTOPE_OPTIONS`。API 方法
`CyESIData.deisotope()` 的默认值如下：

| 参数 | 默认值 | 含义 | 推荐与注意事项 |
|---|---:|---|---|
| `isotope_diff` | `1.003355` | 单电荷碳-13 同位素质量差 | 常规单电荷碳同位素保持默认 |
| `ppm_tol` | `1.0` | 同位素质量差匹配容差 | 先用 1 ppm；仪器误差更大时谨慎增加 |
| `max_isotope_order` | `3` | 检查 M+1 至 M+n 的最大阶数 | 常规代谢组 2–3；增大显著增加候选关系 |
| `r_square_threshold` | `0.95` | 母峰与候选同位素峰共变回归阈值 | 0.90–0.99；越高越保守 |
| `carbon13_abundance` | `0.0109` | 碳-13 自然丰度 | 天然丰度样品保持默认；同位素示踪实验不应直接套用 |
| `intensity_threshold` | `0.0` | 回归时视为缺失的低强度上限 | 有明确检出限时设置，否则保持 0 |
| `safety_factor` | `1.0` | 理论同位素强度上限的放宽系数 | 大于 1 更宽松；修改前应检查 `final_table` |
| `merge_mode` | `keep_parent` | `keep_parent` 保留母峰原强度；`sum` 把子峰加到母峰 | 定量分析通常先用 `keep_parent` |
| `remove` | `True` | 是否从矩阵删除判定的同位素子峰 | 想审计但不删除时设为 `False` |
| `inplace` | `True` | 是否直接修改对象 | `False` 返回完整判定结果字典，适合调参审计 |

去同位素依赖跨细胞强度关系。细胞数很少、存在共洗脱共调控峰或进行稳定同位素示踪时，结果
需要特别谨慎。

### 异常值过滤

Notebook 默认关闭，开启后调用 scikit-learn `IsolationForest`：

```python
RUN_OUTLIER_REMOVAL = True
OUTLIER_OPTIONS = dict(contamination="auto", random_state=42)
```

- `contamination="auto"` 让算法使用内置阈值；已知异常比例时可设 0–0.5 之间的小数。
- 始终固定 `random_state` 以保证可重复。
- 先查看 PCA/UMAP 和质控指标，再决定是否删除异常细胞。

### 缺失值填补

| Notebook 参数 | 默认值 | 选项 | 推荐 |
|---|---:|---|---|
| `IMPUTE_METHOD` | `None` | `knn`、`mean`、`median`、`most_frequent` | 默认不填补；确认 0 表示缺失而非真实低信号后再开启 |
| `IMPUTE_OPTIONS` | `n_neighbors=5` | 传给相应 scikit-learn imputer | KNN 可从 5 开始；细胞少时不得超过有效样本规模 |

API 默认是 `impute(method="knn", missing_values=0)`。所有方法都会修改当前对象，并保留原行列索引。

### 归一化

Notebook 和 API 的默认方法都是 `total`；Notebook 默认额外传入 `scale=1.0`。

| 方法 | 主要参数及默认值 | 含义 | 使用建议 |
|---|---|---|---|
| `total` | `scale=1.0` | 每个细胞除以行总强度再乘 scale | 单细胞总信号差异明显时的首选起点 |
| `max` | `axis=1` | 除以每个细胞最大值 | 由稳定强峰主导且关心相对峰型时使用 |
| `quantile` | 无 | 使每个特征列具有共同分位数分布 | 假设多数特征分布相近；可能削弱真实全局差异 |
| `pqn` | `reference="median"` | 概率商归一化 | 代谢组常用；也可用 `mean` 或自定义一维参考谱 |
| `zscore` | `axis=0` | 按特征做标准化 | 适合降维/聚类，不再保留原始强度尺度 |
| `log` | `pseudo=1e-6` | 计算 `log1p(X+pseudo)` | 压缩高强度长尾；通常应在尺度归一化后单独评估 |
| `minmax` | `axis=0` | 每个特征缩放到 0–1 | 适合部分机器学习输入，不适合解释绝对倍数 |

注意：当前接口一次只执行一种方法。若需要“总量归一化后再 log”，必须连续调用两次，并记录顺序。

## SDF 注释

| Notebook 参数 | API 参数 | 默认值 | 含义与推荐 |
|---|---|---:|---|
| `SDF_PATH` | `sdf_path` | `None` / API 必填 | LIPID MAPS 风格 SDF；`None` 关闭注释 |
| `ANNOTATION_PPM` | `ppm_tol` | `5.0` / API 必填 | 候选理论 m/z 容差；校准良好时 3–5 ppm |
| `ION_MODE` | `search_mode` | `pos` | `pos`、`neg` 或 `both`；应与采集模式一致 |
| `MAX_ANNOTATIONS_PER_MZ` | `max_results_per_mz` | `5` / API `None` | 每个 m/z 最多保留候选数；探索时可增加 |

自定义加合物可通过 API 的 `adducts_pos`、`adducts_neg` 传入。候选按绝对 ppm 误差排序。

## Notebook：分析参数

| 参数 | 默认值 | 含义 | 推荐与限制 |
|---|---:|---|---|
| `EXPORT_SUMMED_SPECTRUM` | `False` | 为原始目录输出总谱 mzML | 只在检查公共峰或归档时开启，增加一次完整汇总计算 |
| `RUN_EMBEDDING` | `True` | 运行 PCA 和 UMAP | 聚类、轨迹和比值 UMAP 图依赖它 |
| `PCA_COMPONENTS` | `50` | 请求的主成分数 | 实际自动截断到 `min(细胞数, 特征数)`；常用 20–100 |
| `UMAP_NEIGHBORS` | `15` | UMAP 近邻数 | 10–30 强调局部结构，较大值强调全局连续性；必须小于细胞数 |
| `UMAP_MIN_DIST` | `0.7` | UMAP 最小距离 | 0.1 产生紧凑簇；0.5–0.8 更适合观察连续轨迹 |
| `RANDOM_STATE` | `42` | PCA/UMAP 随机种子 | 固定以保证复现 |
| `RUN_CLUSTERING` | `False` | 是否进行细胞聚类 | 先检查 UMAP 和批次效应后开启 |
| `CLUSTER_METHOD` | `leiden` | `leiden` 或 `louvain` | Leiden 通常是首选；两者依赖不同可选包 |
| `CLUSTER_RESOLUTION` | `1.0` | 社区划分分辨率 | 0.2–2.0 调试；越高通常簇越多 |
| `PARAMETERIZATION_KEY` | `time` | 时间/伪时间所在的 `obs` 列 | 原始处理自动创建 `time`；自定义真实时间需先写入 `peak_meta` |
| `RUN_TRAJECTORY` | `True` | 在 UMAP 上做滑动窗口轨迹 | 需要 UMAP 和有限的参数化列 |
| `RUN_METABOLIC_VELOCITY` | `False` | 计算局部特征变化率及总体速度 | 参数化列必须有变化；至少两个细胞 |
| `RUN_METABOLITE_TRENDS` | `True` | 计算特征随时间的池化趋势与显著性 | 大数据可能耗时；默认绘制排名靠前特征 |
| `RUN_TREND_CLUSTERING` | `False` | 对代谢趋势曲线聚类 | 需要先运行趋势分析和聚类依赖 |
| `WINDOW_SIZE` | `1000` | 每个滑动窗口的细胞数 | Notebook 会截断到总细胞数；不是秒、分钟或时间比例 |
| `STEP_SIZE` | `300` | 相邻窗口起点相隔的细胞数 | 小于窗口可形成平滑重叠；Notebook 会限制为 1 到窗口大小 |
| `PLOT_TOP_N` | `30` | 趋势热图和趋势聚类最多使用的特征数 | 20–100；注释名称为空的特征会在命名热图中跳过 |
| `RUN_STABILITY_QC` | `False` | 生成时间分箱稳定性图 | 原始数据初次质控时建议开启 |
| `TIME_BINS` | `12` | 稳定性分析时间分箱数 | 保证每箱有足够细胞；细胞少时降低 |
| `MONITOR_MZ` | `REF_MZ` | 稳定性箱线图监测离子 | 可改为内标或已知稳定代谢物；匹配最近的现有特征 |
| `RATIO_MZ` | `None` | 两个离子的比值 UMAP | 例如 `(768.5903, 774.6377)`；分母为 0 时记为 NaN |

## CLI 参数

```text
uv run --locked scmm-process INPUT OUTPUT --ref-mz REF_MZ [options]
```

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `INPUT` | 必填 | 单个 mzML/mzXML 或原始文件目录 |
| `OUTPUT` | 必填 | 保存根目录；内部还会创建数据集名目录 |
| `--ref-mz` | 必填 | 参考离子 m/z |
| `--ppm-tol` | `10.0` | 对齐容差 ppm |
| `--resolution` | `35000.0` | m/z 200 分辨率 |
| `--cell-snr` | `5.0` | 细胞检测阈值 |
| `--peak-snr` | `3.0` | 细胞内特征检测阈值 |
| `--jobs` | `-1` | 目录模式并行数 |
| `--overwrite` | 关闭 | 允许覆盖同名标准结果文件 |
| `--verbose` | 关闭 | 启用 DEBUG 日志 |

CLI 只负责原始数据预处理和保存，不执行归一化、去同位素或下游分析。需要这些步骤时使用
Notebook 或 Python API。

## 特征轴合并参数

`CyESIData.alignwith()` 和批处理 `concat()` 使用：

| 参数 | 默认值 | 含义 | 推荐 |
|---|---:|---|---|
| `ppm_tol` | `5.0` | 两数据集特征视为同一 m/z 的容差 | 使用质量校准表现确定，通常 3–10 ppm |
| `mz_merge_options` | `union` | `union` 保留双方未匹配峰；`ref` 只保留参考数据集特征 | 探索分析用 `union`，固定面板或严格比较用 `ref` |
| `ref_idx` | `0` | `concat()` 中作为参考轴的数据集编号 | `ref` 模式下选择质量和覆盖最可靠的数据集 |

合并会原地修改参考对象；未匹配位置填 0。批次校正不是该步骤的一部分。

下一步：[Python API](04-python-api.md) 或 [分析方法](05-analysis.md)。
