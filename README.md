# scMM

`scMM` 是用于处理和分析 CyESI 单细胞质谱数据的 Python 包。它可以从
mzML/mzXML 原始谱构建“细胞 × m/z 特征”矩阵，并完成细胞事件识别、谱峰对齐、
归一化、缺失值填补、去同位素、精确质量注释、降维聚类、时间轨迹和代谢趋势分析。

## 从这里开始

- [安装与环境配置](docs/01-installation.md)
- [参数化 Notebook 工作流](docs/02-notebook-workflow.md)
- [参数默认值、含义与推荐](docs/03-parameters.md)
- [Python API 与批处理](docs/04-python-api.md)
- [降维、聚类与时间分析](docs/05-analysis.md)
- [数据模型与输出文件](docs/06-data-and-output.md)
- [常见问题与参数调优](docs/07-troubleshooting.md)
- [开发与质量检查](docs/08-development.md)
- [实验室网页 UI 与部署](docs/09-web-ui.md)
- [完整文档索引](docs/README.md)

## 最短使用路径

创建环境并安装项目：

```bash
uv sync --locked --all-extras --dev
```

该命令根据 `pyproject.toml` 和已提交的 `uv.lock` 创建项目专用 `.venv`，并以可编辑模式安装
scMM。后续命令统一通过 `uv run` 执行，无需手工激活环境。

打开 [scMM_workflow.ipynb](scMM_workflow.ipynb)，至少修改：

```python
INPUT_PATH = Path("data/sample.mzML")
OUTPUT_ROOT = Path("results")
REF_MZ = 734.5929  # 必须换成实验使用的参考离子
```

然后执行 `Run All`。Notebook 可以自动识别单个原始文件、原始文件目录和已有处理结果目录。

也可以直接使用命令行：

```bash
uv run --locked scmm-process input.mzML results --ref-mz 734.5929
uv run --locked scmm-process raw-data/ results --ref-mz 734.5929 --jobs 4
```

或使用 Python API：

```python
from scMM.file.data import CyESIData

data = CyESIData.load_from_file(
    "sample.mzML",
    ref_mz=734.5929,
    cell_snr=5.0,
    peak_snr=3.0,
)
data.normalize("total")
result_dir = data.save("results")
```

重新载入时必须传入实际的数据集目录，而不是它的父目录：

```python
reloaded = CyESIData.load_from_processed(result_dir)
```

在本机启动引导式网页，用服务器已经挂载的目录直接选择、预览并处理原始数据：

```bash
uv run --locked scmm-ui \
  --storage "原始数据=/mnt/ms-data" \
  --output "处理结果=/mnt/scmm-results" \
  --address 0.0.0.0 \
  --port 5006
```

页面支持 TIC/EIC/合并谱预览、处理参数预检、断开浏览器后继续运行的后台任务、PCA/UMAP
质量检查以及标准结果和质量表下载。局域网或 Tailscale 访问方式和目录边界配置见
[实验室网页 UI 与部署](docs/09-web-ui.md)。

## 支持范围

- Python：3.11–3.12；`.python-version` 将本地开发默认固定为 Python 3.12。
- 原始谱：mzML、mzXML。
- 核心依赖：PyOpenMS、NumPy、pandas、SciPy、scikit-learn、AnnData。
- 可选分析：Matplotlib、Seaborn、UMAP、Palantir、Leiden/Louvain。
- 网页 UI：Panel、Plotly、UMAP；执行完整同步时已包含，也可单独使用
  `uv sync --locked --extra ui`。

项目当前版本为 `0.2.0`，开发状态为 Alpha。正式分析前应使用标准样品和实验质控数据验证
参考离子、ppm 容差、SNR 阈值及归一化方案。
