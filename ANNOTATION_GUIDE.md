# SCMM 物质注释 / Compound annotation

启动网页后，在左侧选择“物质注释”。该页面复用已有 Python 环境；Thermo RAW 文件沿用安装脚本提供的官方 RawFileReader 插件。已经安装 SCMM 的电脑可覆盖更新源码，运行一次 `setup_scmm.cmd`，补齐所缺的 `pygoslin`、Excel 读取包及集合图所需的 `matplotlib-venn`、`upsetplot`，再使用原来的 `start_scmm.cmd` 启动。当前开发环境已安装这些依赖。

## 操作流程

1. **同位素识别**：调用 `CyESIData.deisotope`，参数名分别对应 `isotope_diff`、`ppm_tol`、`max_isotope_order`、`r_square_threshold`、`carbon13_abundance`、`intensity_threshold`、`safety_factor`。固定 `remove=False`、`merge_mode='keep_parent'`，保留矩阵和各同位素强度；使用 `raw` 层（存在时）与当前有效细胞。零值作为缺失值。默认间隔适用于单电荷。原实现包含特征两两矩阵，运行内存随特征数平方增长。
2. **MS1 数据库**：选择本机文件或上传数据库，设置正/负离子模式、加合物与 ppm。支持 CSV、TSV、LIPID MAPS SDF 和下述 JSON 记录。
3. **MS/MS 匹配**：可一次上传多个 DDA/PRM 文件，格式为 Thermo RAW、mzML、mzXML；也可从本机选择文件。标准谱库支持 MSP、MGF、JSON、CSV/TSV。设置前体/碎片容差、处理范围、Profile 重采样间隔、去基线宽度、质心化信噪比、相对强度阈值、最少碎片数与相似度阈值。
4. 点击 **开始注释 / 刷新**，查看结果页。每个候选单独一行，滚动时按需传输约 100 行。勾选接受或使用 **Accept All** / **Accept First**。文本修改后按 Enter 或移开焦点保存。
5. 在 **选择注释写入字段与 var 键名** 表格中，勾选名称、分子式、类别、质量误差、相似度等需要保存的指标，逐行指定目标键名，再点击 **写入 var**。默认仅选名称，目标键为 `annotation`。完整候选、接受状态、参数与用于镜像图的谱图一起保存在 H5AD 的 `uns['annotation']`。

同位素行始终排在末尾，通过 `annotation_isotope` 的 uint8 标记管理。在表格中取消“同位素”或在识别页恢复所选特征，再点击刷新即可重新匹配。**去同位素** 写入可逆的 `var['feature_keep_annotation_isotope']`（0 隐藏，1 保留），可在预处理页停用或恢复。同位素特征跳过数据库搜索、接受、写入 var 与注释 CSV 导出。已有其它特征筛选标签继续生效。

刷新会重新计算候选并清空旧候选的手动编辑及接受状态；需要保存这些调整时，先下载注释表或导出 H5AD。修改分子式、加合物等文本用于结果整理，评分保留计算时的数值；要重新评分请更新数据库并刷新。

## 手动注释导入

在“5 · 手动注释”中选择或上传 CSV、TSV、XLSX、XLS 文件，设置表头与 Excel 工作表后点击“读取注释表”。在字段映射表中勾选要导入的列，并分别填写目标 var 键名，可一次导入名称、类别、分子式等多个字段。每行一个注释字段的表格可先转置，再在映射表中选择对应字段。

“自动识别”优先通过导入表的键值与 var 元信息对齐；具有 `feature_id` 的表格可按 var 索引匹配，`mz`、`m/z`、`observed_mz` 按指定 ppm 匹配。按顺序导入时，记录数需等于 var 长度。也可选择“按键值匹配”，明确指定两侧字段。同一匹配键的多个名称用 `/` 合并。

“导入记录行号”按当前表格方向从 1 开始编号，支持 `1-5,8,10-12`，留空选择全部记录。按顺序对齐时，行号继续对应原始 var 位置。检查前 100 条匹配预览和匹配数量后，点击“写入手动注释”。名称默认写入 `manual_annotation`，其它字段的默认目标键与来源字段同名。未匹配特征保留目标列已有值，“未匹配特征的注释置空”可将这些位置清空。写入的数据随 H5AD 保存。

数据库注释写入仅使用已接受且具有名称的候选，同位素特征在该步骤跳过。同一特征的多个不同值用 `/` 合并。数值指标在每个特征均至多有一个不同数值时保留数值类型，缺失位置为 NaN；文本及多值指标使用文本，缺失位置为空。各字段的目标键名需分别填写，全部字段确认后统一写入。

## 细胞帧叠加谱辅助注释

在“4 · 注释结果”的注释表格与饼图之间，明确多选参与叠加的质谱源文件，并核对每个文件的原始路径。谱图位于上方，文件选择和参数设置位于下方。路径可通过“选择文件”按钮指定。抽样范围限定在这些文件对应的细胞帧，默认对每个选中文件分别随机抽取 20 个有效细胞帧后合并，随机种子为 42。抽样口径也可改为“所选文件合计”。帧数、m/z 范围、分辨率和采样点数均可调整。

点击“读取并叠加细胞帧”后，界面列出实际文件路径、逐文件帧数和原始 `frame_id`，帧序号从 0 开始。读取沿用现有 RAW、mzML、mzXML 接口，按原始帧序号定位并核对 MS1 级别。

点击候选表中固定在左侧的 Feature ID，或使用“谱图定位特征”，即可局部放大并高亮特征峰及关联同位素峰。切换特征复用已叠加的谱图。修改文件、抽样参数或有效细胞范围后，点击叠加按钮生成对应的新谱。图形支持现有 Matplotlib 预览与 SVG/PNG 保存，各交互图的缩放和平移范围限制在完整数据的绘图范围内。

新增参数的默认值和帮助文本位于 `scMM/config/defaults.json` 的 `annotation.manual_*` 和 `annotation.cell_*` 项。m/z 上下限设为 `null` 时根据特征范围自动填入。

## 按 obs 分组的集合图与类别组成

在“预处理与质量控制”和“物质注释”中均可使用“分组集合关系”。选择任意 obs 分组键，如 `source_file`、实验组别或疾病分类，再多选需要比较的组别。特征集合依据当前有效细胞和有效特征，在指定数据层中计算；默认选择 raw 层，有限强度值大于 0 且组内至少一个细胞检出即计入集合。可提高强度下限和组内最低检出率。

Venn 图展示 2–3 组的重叠，UpSet 展示至少 2 组的交集数量和成员矩阵。比较注释时，将“集合内容”设为“注释物质”，选择“var 注释列”来源，再在“表示注释结果的 var 键”中指定所需字段，如 `identification`、`annotation` 或自定义键名。支持文本、分类和数值标识，空值跳过。同一组中相同注释值去重；切换键名后点击“计算分组集合”更新图形。也可使用候选表中已接受的结果。var 中以 `/` 连接的多个名称可拆分统计；关闭拆分时，整段文字作为一个物质名称。同位素从物质集合统计中排除。

在物质注释页勾选“展示各组物质种类组成”，可按候选表类别或指定 var 类别列绘制堆叠柱状图，并切换物质数量与百分比。每组按名称与类别去重，同一物质的不同类别归属分别计数。点击“计算分组集合”生成统计，集合成员及类别数量均可导出 CSV。

注释集合仅使用具有有效注释的主峰及独立特征。空白、缺失值和 `NaN`、`None`、`unknown` 等空注释占位文字在统计前清理。当前 `annotation_isotope` 标记为同位素的特征予以过滤；旧数据可使用 `deisotope_role` 或 `isotope_order` 确定同位素。手动恢复以当前 `annotation_isotope` 标记为准。Venn、UpSet、集合成员 CSV 和类别组成使用同一套过滤结果。

集合图与组成图使用 Matplotlib 绘制。“Matplotlib 绘图参数与导出”弹窗支持尺寸、字体、字号、cmap、透明度、线宽及 DPI 设置。点击“预览”检查图片，再点击“下载”选择保存位置。SVG 保留原始文字对象。相关默认参数位于 `defaults.json` 的 `sets.*` 项。

## 数据库格式

MS1 CSV 最简示例：

```csv
name,formula,neutral_mass,category,database_id
Glucose,C6H12O6,180.063388,Carbohydrate,example_1
```

`neutral_mass` 为中性单同位素质量，可由 `formula` 自动计算；可以同时提供。名称也兼容 `COMMON_NAME`、`ABBREVIATION`，质量兼容 `EXACT_MASS`，类别兼容 `CATEGORY`、`MAIN_CLASS`。分子式用于计算真实元素组成的自然同位素分布。

同位素标记分子式支持 `C15[13C]3H34O5`、OpenMS 的 `C15(13)C3H34O5` 和含氘的 `C20H21D11O3` 写法。计算保留固定同位素的原子数与质量，结果表保留库文件中的原始分子式。加合物别名 `[M+H-H2O]+`、`[M.Cl]-`、`[M.]+` 分别映射到 `[M-H2O+H]+`、`[M+Cl]-`、`[M]+`。

MS/MS JSON 示例：

```json
[
  {
    "name": "Example compound",
    "formula": "C6H12O6",
    "precursor_mz": 181.070664,
    "adduct": "[M+H]+",
    "polarity": "pos",
    "category": "Carbohydrate",
    "peaks": [[61.0284, 100], [91.0390, 65], [119.0340, 35]]
  }
]
```

指定 `precursor_mz` 时同时提供 `adduct`。`peaks` 为 `[m/z, intensity]` 列表；CSV 中使用 JSON 字符串或 `61.0284:100;91.0390:65`。MSP 使用 `Name`、`Formula`、`PrecursorMZ`、`Precursor_type`、`Ion_mode`、`Num Peaks` 及峰列表；MGF 使用 `TITLE`、`PEPMASS`、`ADDUCT`、`FORMULA`。未提供中性质量或分子式的谱库条目，需要实际 MS/MS 证据达到阈值后才报告为候选。

支持加合物与质量偏移（Da）：

| 加合物 | 质量偏移 |
|---|---:|
| [M]+ | -0.000548580 |
| [M+H]+ | +1.007276467 |
| [M+Na]+ | +22.989220702 |
| [M+K]+ | +38.963157906 |
| [M+NH4]+ | +18.033825553 |
| [M-H2O+H]+ | -17.003288217 |
| [M+Li]+ | +7.015454857 |
| [M-H]- | -1.007276467 |
| [M+HCOO]- | +44.998202837 |
| [M+CH3COO]- | +59.013852901 |
| [M+Cl]- | +34.969401262 |

`[M]+` 表示失去一个电子的分子离子。`[M+FA-H]-` 作为甲酸根加合物的旧名称继续兼容。ppm error 定义为 `(observed / theoretical - 1) × 10⁶`。

## 评分、图表与导出

- 每个特征在全部实测文件中按前体 ppm 收集 **MS2** 扫描，保留“文件名:原生 scan ID”。Thermo RAW 通过官方扫描事件获取前体质量，MS2 优先直接使用官方 RawFileReader 的质心流，按 ppm 汇总；常规 MS1 读取保留原有谱型。其它 Profile 输入沿用插值叠加、top-hat 去基线与 OpenMS PeakPickerHiRes 质心化。相同 m/z 的不同保留时间和碰撞能量扫描共同汇总。
- 碎片采用 ppm 内最近的一对一匹配；dot product 为平方根强度向量的归一化余弦，未匹配离子进入分母。满足最少匹配碎片数及相似度阈值时记为 MS/MS 匹配，否则有质量证据的候选记为 MS1。
- 同位素理论分布使用 OpenMS `EmpiricalFormula` 和 `CoarseIsotopePatternGenerator`，包含加合物/中性丢失的元素组成。实测分布使用 `deisotope` 回归的 M+k/母峰强度比例。没有已识别同位素时显示空值。综合分数为可用证据按设定权重计算的平均值，两项权重均为零时综合分数为零。
- 默认排序：同位素标记、observed m/z、feature id 升序，特征内部综合分数降序，随后按绝对 ppm error 升序。PC、PE、Cer、DGTS 等规则脂质转换到总组成层级；其它物质逐项使用原名。原始候选及碰撞能量谱的指标保存在 `candidate_metrics` 中。
- **MS/MS 对比** 打开弹窗，共用 m/z 横轴，实测向上、标准向下，分别归一化到 100%，图例区分。支持 Matplotlib 预览、线宽/字号/尺寸/DPI 调整及 PNG/SVG 下载；SVG 保留原始文字。
- CSV 导出排除同位素及被其它标签隐藏的特征。规则脂质按总组成输出，复杂色素、天然产物、衍生物、小分子分别以原名逐行输出。原始名称放在 `original_names`，代表谱原始名称放在 `original_name`，各候选完整指标放在 `candidate_metrics`。
- 饼图按独立特征统计 MS/MS、MS1 和未匹配比例；类别图按已接受注释的特征-类别统计。同一特征接受多个类别时可贡献多个类别计数。
- “标准化脂质名称”可将已有结果重新合并到总组成层级；写入 var 和导出 CSV 也应用同一规则。相同总组成的标准谱按 **MS/MS dot product 降序、匹配碎片数降序** 选择，继续相同时参考综合分数及绝对 ppm 误差。表格指标和镜像谱始终对应同一张代表谱。原始库名标示谱图来源。详细规则、示例与依据见 [LIPID_NOMENCLATURE.md](LIPID_NOMENCLATURE.md)。

参数默认值与悬停说明位于 `scMM/config/defaults.json` 的 `annotation.*` 项。

参考接口：[OpenMS 同位素分布](https://pyopenms.readthedocs.io/en/release-3.4.1/user_guide/chemistry.html)、[Thermo RawFileReader 官方示例](https://github.com/thermofisherlsms/RawFileReader/blob/main/ExampleProgram/NetCore/Program.cs)、[LIPID MAPS 简写命名规则](https://www.lipidmaps.org/lipid_nomenclature)。

## English quick start

Open **Compound annotation / 物质注释** in the sidebar. Run the existing deisotope method, select an MS1 database and adducts, optionally upload several RAW/mzML/mzXML MS2 acquisitions and MSP/MGF/JSON libraries, then run or refresh annotation. Review editable candidates in the virtual table. Accept all candidates or the first ranked candidate per feature, inspect mirror spectra, write accepted names to `var`, and download the complete CSV. Isotope features remain reversible through uint8 flags and appear at the end of the table. Refresh recalculates candidates and resets previous candidate edits and acceptance; export them first when needed. The existing environment and launcher remain usable.

The **Manual annotation** tab imports CSV/TSV/Excel fields by feature order, feature ID, var metadata, or m/z within a chosen ppm tolerance. Select source columns and target var keys in the mapping table. Optional one-based row ranges such as `1-5,8` select individual records. The library-results mapping table also supports multiple annotation fields. Preview the alignment and commit the selected fields together.

In **Annotation results**, explicitly select several original MS files for the cell-frame spectrum. The default samples up to 20 active cell frames per selected file, then sums them. A total-frame sampling option is also available. The report lists each selected file and its zero-based frame IDs. Click a Feature ID to zoom to its peak and associated isotopes. Changing the feature reuses the summed spectrum.

Both QC and annotation pages provide Venn and UpSet plots grouped by an obs key. Choose features or annotated compound names, detection thresholds, and either accepted candidates or a var annotation column. The annotation page also provides grouped compound-class composition with counts or percentages. Set members and class counts can be exported to CSV. Matplotlib dialogs preview and save SVG/PNG files, with editable SVG text.
