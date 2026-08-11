# 开发与质量检查

[返回文档索引](README.md) · [安装说明](01-installation.md)

## 代码结构

```text
scMM/
├── cli.py                 # scmm-process 命令行入口
├── file/
│   ├── io.py              # mzML/mzXML 文件边界与稳定导出
│   ├── _spectrum.py       # Orbitrap 网格、谱汇总与峰细化
│   ├── _alignment.py      # 峰到目标 m/z 的匹配与帧聚合
│   ├── data.py            # CyESIData 稳定门面与构造入口
│   ├── _dataset_loading.py # 已处理/原始数据装载与组合
│   ├── _dataset_processing.py # 预处理、变换与数据集合并
│   ├── _dataset_interop.py # 保存、访问、注释与 AnnData 转换
│   ├── _deisotope.py      # 去同位素的纯计算、分配与元数据构建
│   └── batch.py           # 独立批处理与结果合并
├── util/
│   ├── peak.py            # 局部谱统计与细胞事件窗口归约
│   ├── normalize.py       # 归一化注册表和内置方法
│   ├── annotation.py      # SDF 读取与稳定搜索门面
│   ├── _adducts.py        # 加合物定义与质量换算
│   ├── _annotation_search.py # 候选生成、排序与结果模式
│   └── denoise.py         # 矩阵分解与峰重建工具
└── plot/
    ├── engine.py          # PlotEngine 共享状态与领域能力组合
    ├── _engine_*.py       # 降维、轨迹、聚类和特征网络领域能力
    ├── _trajectory.py     # Palantir、窗口轨迹、速度和趋势统计
    ├── _trend_clustering.py # 趋势距离与聚类算法
    ├── embedding.py       # 轻量降维接口
    └── msplot.py          # EIC、谱和调试图
```

测试位于 `tests/`，覆盖数据保存/加载、对齐、去同位素、归一化、谱 I/O、绘图、轨迹和 CLI。

## 开发环境

```bash
conda env create -f environment.yml
conda activate scmm-dev
python -m pip install --no-deps -e .
```

更新环境：

```bash
conda env update -f environment.yml --prune
python -m pip install --no-deps -e .
```

## 完整验证

从仓库根目录运行：

```bash
ruff format --check .
ruff check .
pytest -W error
python -m build
```

`pytest -W error` 会把警告提升为错误，有助于尽早发现 pandas、NumPy、scikit-learn 或 PyOpenMS
升级引入的兼容问题。

只运行相关测试：

```bash
pytest tests/test_data.py -q
pytest tests/test_io.py -q
pytest tests/test_trajectory.py -q
```

覆盖率：

```bash
pytest --cov=scMM --cov-report=term-missing
```

## Notebook 检查

通用 notebook 应满足：

- 所有用户路径和实验参数集中在参数单元。
- 不包含个人主目录绝对路径。
- 提交前清除大量执行输出和临时图。
- 参数单元带 `parameters` 标签，便于 Papermill 等工具注入配置。
- 示例默认值不执行不可逆或高风险步骤。
- 文档中说明无法仅凭示例默认值确定的实验参数，尤其是 `REF_MZ`。

基本结构和语法检查可以在没有科学计算依赖时完成：

```bash
python -m json.tool scMM_workflow.ipynb >/dev/null
python - <<'PY'
import ast
import json

with open("scMM_workflow.ipynb", encoding="utf-8") as handle:
    notebook = json.load(handle)

for number, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code":
        ast.parse("".join(cell["source"]), filename=f"cell-{number}")
print("notebook syntax OK")
PY
```

这不能代替在 `scmm-dev` 环境中使用代表性 mzML 执行整个流程。

## API 设计约定

- 高层可变换方法通常原地修改 `CyESIData` 并返回自身，以支持方法链。
- 保存方法返回实际创建的路径。
- 公开入口应验证维度、范围和有限数值，并给出明确异常。
- `data`、`peak_meta` 和 `feature_meta` 必须保持行列一一对应。
- 随机算法公开 `random_state`/`seed`。
- 可选绘图或聚类依赖应在调用对应功能时才导入并给出针对性提示。
- `CyESIData` 只负责容器状态和处理溯源；较长的数值流程应拆到相邻的私有模块，并优先实现为
  不修改输入的纯函数。`_deisotope.py` 是这一边界的参考：公开方法组装参数并提交结果，候选检测、
  回归、筛选、分配和元数据生成各自独立。
- 数据容器的新能力应归入装载、处理或互操作领域之一；`data.py` 只组合这些能力并维护稳定的
  构造入口。跨数据集合并应先生成完整 `DatasetState`，确认成功后再一次性更新当前对象。
- 峰处理流程应把局部统计、事件分段和结果组装分开；并行执行的单事件函数应保持为模块级纯函数，
  便于独立验证且避免闭包携带整个调用上下文。

## 添加归一化方法

归一化方法通过注册表扩展：

```python
from scMM.util.normalize import register_norm


@register_norm("custom")
def norm_custom(X, params):
    scale = params.get("scale", 1.0)
    return X * scale
```

新方法应验证二维输入、零分母、NaN/Inf 和参数类型，并在 `tests/test_normalize.py` 添加测试。

## 添加降维方法

轻量接口使用 `register_dim`：

```python
from scMM.plot.embedding import register_dim


@register_dim("custom")
def run_custom(X, params):
    model = CustomModel(**params)
    return model.fit_transform(X)
```

输出必须是每个细胞一行、至少两列的二维数组。

## 数据与 Git

- 不提交原始 mzML/mzXML、处理矩阵、H5AD 或大体积执行输出，除非仓库策略明确允许。
- 测试数据应尽量使用代码构造的小型合成谱和矩阵。
- 不在 notebook 或源码中写个人绝对路径。
- 提交信息建议使用项目现有风格，例如 `docs: ...`、`refactor: ...`、`test: ...`。
- 提交前查看 `git status` 和 `git diff --check`，避免把无关本地修改带入提交。

## 发布前检查

```bash
python -m build
python -m pip install --force-reinstall dist/*.whl
scmm-process --help
pytest -W error
```

还应在受支持的 Python 版本和至少一个代表性 mzML/mzXML 文件上完成端到端验证。
