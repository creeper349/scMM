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
- [完整文档索引](docs/README.md)

## 最短使用路径

创建环境并安装项目：

```bash
conda env create -f environment.yml
conda activate scmm-dev
python -m pip install --no-deps -e .
```

打开 [scMM_workflow.ipynb](scMM_workflow.ipynb)，至少修改：

```python
INPUT_PATH = Path("data/sample.mzML")
OUTPUT_ROOT = Path("results")
REF_MZ = 734.5929  # 必须换成实验使用的参考离子
```

然后执行 `Run All`。Notebook 可以自动识别单个原始文件、原始文件目录和已有处理结果目录。

也可以直接使用命令行：

```bash
scmm-process input.mzML results --ref-mz 734.5929
scmm-process raw-data/ results --ref-mz 734.5929 --jobs 4
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

## 支持范围

- Python：3.11–3.13；推荐使用 `environment.yml` 固定的 Python 3.12。
- 原始谱：mzML、mzXML。
- 核心依赖：PyOpenMS、NumPy、pandas、SciPy、scikit-learn、AnnData。
- 可选分析：Matplotlib、Seaborn、UMAP、Palantir、Leiden/Louvain。

项目当前版本为 `0.2.0`，开发状态为 Alpha。正式分析前应使用标准样品和实验质控数据验证
参考离子、ppm 容差、SNR 阈值及归一化方案。
