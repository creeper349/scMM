# scMM 文档索引

本文档按实际使用顺序组织。首次使用建议依次阅读 1–3；需要脚本化或深入分析时再阅读后续章节。

## 入门

1. [安装与环境配置](01-installation.md)
   - Conda 环境、可编辑安装、Jupyter 内核、可选依赖和安装验证。
2. [参数化 Notebook 工作流](02-notebook-workflow.md)
   - 输入准备、运行顺序、功能开关、输出位置和典型配置。
3. [参数默认值、含义与推荐](03-parameters.md)
   - Notebook、CLI、原始谱预处理、去同位素、注释和分析参数的完整参考。

## 使用与分析

4. [Python API 与批处理](04-python-api.md)
   - 单文件、多文件、已有结果、批量独立处理、数据集合并和方法链。
5. [降维、聚类与时间分析](05-analysis.md)
   - AnnData、PCA、UMAP、Leiden/Louvain、Palantir、轨迹、代谢速度和趋势聚类。
6. [数据模型与输出文件](06-data-and-output.md)
   - `CyESIData` 的四类数据、保存目录、字段语义和可重复性建议。

## 维护与排错

7. [常见问题与参数调优](07-troubleshooting.md)
   - 未检出细胞、特征过多/过少、内存、重复保存、依赖错误和分析异常。
8. [开发与质量检查](08-development.md)
   - 测试、格式检查、构建、代码结构和贡献前检查。
9. [实验室网页 UI 与部署](09-web-ui.md)
   - 挂载目录选择、TIC/EIC/合并谱预览、后台处理、质量检查、结果下载和 Tailscale 访问。

## 快速选择入口

| 目标 | 推荐入口 |
|---|---|
| 第一次处理一份数据 | [参数化 Notebook](02-notebook-workflow.md) |
| 在服务器批量预处理 | [CLI 或批处理 API](04-python-api.md) |
| 查询某个参数默认值 | [参数参考](03-parameters.md) |
| 理解生成的 CSV/pickle | [数据与输出](06-data-and-output.md) |
| 调整 UMAP、轨迹或趋势 | [下游分析](05-analysis.md) |
| 处理结果异常或报错 | [排错指南](07-troubleshooting.md) |
| 从浏览器查看并处理服务器原始谱 | [网页 UI](09-web-ui.md) |

## 核心数据流

```text
mzML/mzXML
  └─ 变分辨率 m/z 网格与合并谱
      └─ 合并谱去噪和公共峰提取
          └─ 每个扫描帧对齐到公共 m/z 轴
              └─ 参考离子通道识别细胞事件
                  └─ 细胞 × 特征矩阵
                      ├─ 去同位素 / 异常值 / 填补 / 归一化
                      ├─ SDF 精确质量候选注释
                      └─ PCA / UMAP / 聚类 / 时间轨迹 / 趋势
```

所有高层处理都围绕 `scMM.file.data.CyESIData` 展开；下游分析围绕
`scMM.plot.engine.PlotEngine` 和 AnnData 展开。
