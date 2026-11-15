# SCMM-GUI-0.3.0-fix

内部版本号为 `0.3.0.post1`，用于识别环境升级。发行名称为 SCMM-GUI-0.3.0-fix。

本版改进包括以下内容。

- 修正重复 m/z 特征造成的特征元信息长度不一致。相邻峰按谷底划分质心窗口，保留原有峰宽限制。
- 批量导入支持文件夹及多个 mzML/mzXML/RAW 文件。Windows 使用系统原生文件选择窗口，并提供“选择当前文件夹”按钮。
- 支持逐文件拖动双端滑块选择保留的时间或帧区间。筛选作用于总谱叠加及定量，细胞信息保留原始帧号。
- 自动读取采集参数，支持相关系数分布图与点选名称标注。
- 注释辅助叠加谱加入固定总览，并随局部缩放标记当前范围。
- 改进细胞标记、可逆筛选、分组统计图、显著性标注及图片预览导出。
- 分析任务可在页面切换期间继续运行，处理后的 H5AD 可保存到原文件。
- 整理界面文案，支持中英文切换与扩展语言包。

安装和补丁升级步骤见 `UPGRADE_TO_0.3.0-fix.md`，语言扩展见 `docs/languages.md`。

This update adds valley-bounded centroiding and positional feature metadata filtering, per-file time/frame selection, native batch selection, correlation distribution plots, linked spectrum overviews, improved figure export, background processing, and Chinese/English language packs. The Python package version is `0.3.0.post1`.
