# 单细胞直接进样脂质总组成报告

SCMM 对规则脂质采用 LIPID MAPS **species / sum composition** 层级：类别、总碳数、总双键当量（DBE）、附加氧数，以及固定同位素等信息。DBE 可包含碳碳双键、环、羰基和三键贡献。复杂色素、天然产物、各种复杂衍生物及小分子代谢物逐一保留原始名称。

规则依据：[LIPID MAPS 层级](https://www.lipidmaps.org/lipid_nomenclature/rules/hierarchy)、[通用命名规则](https://www.lipidmaps.org/lipid_nomenclature/rules/general)。

## 名称来源与解析

程序复用 [Goslin Python 参考实现](https://github.com/lifs-tools/pygoslin)（pygoslin 2.2.5），解析 Shorthand、Goslin、LIPID MAPS、SwissLipids、HMDB 等写法，输出 species 名称。优先使用数据库的 `ABBREVIATION`；LMSD 的许多条目已直接提供总组成简写。

MGF 名称中携带的 LM_ID，例如 `Maresin 1 (LMFA04050001)`，会连接同次加载的 LMSD 记录，以使用其简写与类别。原始分子式继续参与质量和同位素计算。分子式中的自然元素和固定同位素计数保持完整。

有中性精确质量或可计算的分子式时，匹配使用它们与加合物计算的理论 m/z；原库 `PEPMASS` 保存为 `library_precursor_mz`。仅提供前体 m/z 的谱库条目沿用该数值。该规则让库内精度较低的前体值仍能关联到正确的精确质量候选。

简化范围采用明确的类别列表，包括 PC、PE、PS、PI、PG、PA 及溶血形式、CL、MG/DG/TG、MGDG/DGDG、SQDG、DGTS/DGTA、Cer、SM、部分简单糖鞘脂、简单游离甾醇和甾醇酯。复杂糖链、氧化甾醇、FA 类信号分子、植物色素、天然产物和其它未纳入类别的记录使用原名，`report_level=unchanged`。例如 Maresin 1、Maresin 2、15-HETE 和叶绿素分别列出。

旧版 H5AD 中已被合并为类别与分子式、或本轮不再简化的总组成行，会从保存的 `candidate_metrics` 恢复原始候选名称。写入 var 时，接受的多个名称继续用 `/` 连接；CSV 中逐个名称列出。

## 总组成规则与示例

| 类型 | 库名称示例 | SCMM 报告名称 |
|---|---|---|
| sn / 链长 / 双键位置异构 | PC 16:0/18:1(9Z)、PC 18:1(9E)/16:0、PC 18:0/16:1 | PC 34:1 |
| 三酰甘油 | TG(16:0/18:1/18:2) | TG 52:3 |
| 心磷脂 | CL(16:0/18:1/18:2/20:4) | CL 72:7 |
| 溶血磷脂 | LPC 0:0/18:1(9Z) | LPC 18:1 |
| 醚脂 | PE O-18:1/18:0 | PE O-36:1 |
| 烯醚脂 | PE P-16:0/20:4 | PE O-36:5 |
| 氧化磷脂 | PC 18:0/18:2;OOH | PC 36:2;O2 |
| 甜菜碱脂质 | DGTS 16:0/18:1 | DGTS 34:1 |
| 复杂脂肪酸衍生物 | FA 18:1;9oxo、15-HETE | 分别保留原名 |
| 神经酰胺 | Cer(t18:0/24:1) | Cer 42:1;O3 |
| 鞘磷脂 | SM(d18:1/24:0) | SM 42:1;O2 |
| 单己糖鞘脂 | GlcCer(d18:1/24:0)、GalCer(d18:1/24:0) | HexCer 42:1;O2 |
| 双己糖鞘脂 | LacCer(d18:1/16:0) | Hex2Cer 34:1;O2 |
| 复杂糖链脂质 | GM3(d18:1/16:0) | 保留原名 |
| 甾醇酯 | CE 18:2、SE 27:2/18:1 | SE 45:3 |
| 游离甾醇 | Cholesterol | ST 27:1;O |
| 磷酸位点异构 | PIP2(3',5')(16:0/18:1) | PIP2 34:1 |
| 固定同位素 | PC 16:0/18:1-d7 | PC 34:1[M[2]H7] |

醚脂 `O-` 保留醚键类型；总组成层级将 `P-` 转为 `O-` 并将烯醚键贡献的 1 个 DBE 计入总数。因此 `PE P-36:4` 与 `PE O-36:5` 具有相同总组成。该处理符合 Goslin 的 species 输出。参考：[LIPID MAPS 简写数据库](https://www.lipidmaps.org/databases/shorthand/overview)、[2020 命名规则论文](https://www.jlr.org/article/S0022-2275%2820%2960017-7/fulltext)。

规则鞘脂保留骨架及脂肪链氧数，Glc/Gal 等单己糖异构归为 Hex，简单双己糖形式归为 Hex2Cer；更复杂的糖链记录保留原名。参考：[鞘脂规则](https://www.lipidmaps.org/lipid_nomenclature/sphingolipids/ceramide)、[糖鞘脂规则](https://www.lipidmaps.org/lipid_nomenclature/sphingolipids/glycosphingolipid)。

甾醇的四环骨架按类别约定处理；`SE 45:3` 为甾醇与酰基的总组成，酯基自身的两个氧和一个 DBE 按约定隐含。程序对 Goslin 2.2.5 保留的甾醇/酰基分隔进一步求和。参考：[LIPID MAPS 甾醇规则及示例表](https://www.lipidmaps.org/lipid_nomenclature/sterols)。

## 合并边界与标准谱选择

合并键为 **feature id + species 名称 + 分子式 + 加合物 + 同位素标记状态**。固定同位素的分子式写法先统一比较；不同脂质类别、不同氧数、不同固定标记或不同加合物分别保留。相同分子式可以对应 PC 与 PE 等不同类别，名称和类别参与报告分组。

每个候选标准谱独立与实测叠加谱匹配。合并时首先选 dot product 最高的候选，相同时选匹配碎片数更多者，再依次比较综合分数和绝对 ppm 误差。相似度、匹配碎片、理论 m/z、ppm 误差、标准谱 ID 和镜像谱均取自这张代表谱。各异构体和不同碰撞能量的原始标准谱保留独立证据，`candidate_metrics` 记录其原始计算结果。

规则脂质名称表示总组成候选，其它物质分别使用原名。`original_name` 是代表谱的库内名称，供追溯标准谱来源；`original_names` 是合并前库名列表，`isomer_count` 是不同库名数量（含同一物质别名的可能）。最佳镜像谱提供当前候选的匹配证据。

接受复选框作用于当前总组成行。写入 var 时，已接受的多个总组成名称通过 `/` 连接。CSV 保留各总组成行与其完整候选指标；H5AD 保存总组成表及全部被命中的标准谱。

## 验证范围

测试覆盖 sn/链组成/双键位置异构、醚脂、氧化脂质、神经酰胺、糖鞘脂、磷酸位点、甾醇酯、固定同位素、标准谱选择、类别保留、CSV/var 写入与 H5AD 往返。

测试包含复杂物质逐项保留、旧版过度简化结果恢复，以及 DGTS 总组成转换。实际 RAW 文件核查见 [xxlz MS/MS 诊断报告](diagnostics/xxlz/REPORT.md)。

## 已安装电脑的更新

覆盖更新源码后运行 `setup_scmm.cmd`，脚本会检查并补齐 pygoslin，再运行 `start_scmm.cmd`。已有 H5AD 的注释结果可以点击“标准化脂质名称”重新合并；要使用 LMSD 的 `ABBREVIATION` 连接和完整候选谱证据，可加载两个谱库后重新点击“开始注释 / 刷新”。
