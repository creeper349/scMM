# 安装与环境配置

[返回文档索引](README.md)

## 推荐方式：uv

先按 [uv 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/) 安装 uv，然后在仓库
根目录同步完整环境：

```bash
uv sync --locked --all-extras --dev
```

uv 会读取 `.python-version`，在需要时安装 Python 3.12，并根据 `uv.lock` 创建项目专用 `.venv`。
项目会以可编辑模式安装，修改本地 `scMM/` 后无需重新安装。不要再用 pip 向 `.venv` 手工添加
长期依赖；发布依赖通过 `uv add` 管理，开发工具通过 `uv add --dev` 管理。

当前正式支持 Python 3.11–3.12。项目把 PyOpenMS 和核心数值栈限制在本机及 CI 已验证的兼容
范围；PyOpenMS 3.3 没有 CPython 3.13 wheel，因此在新 PyOpenMS 版本通过部署机器的 CPU、原始谱
读写和完整测试前，不应只为追新版本移除这些上限。

日常拉取代码后严格复用锁文件：

```bash
uv sync --locked --all-extras --dev
```

只有明确升级依赖时才运行 `uv lock --upgrade`，并将 `pyproject.toml` 与 `uv.lock` 的变更一起审阅、
测试和提交。`uv sync` 默认精确同步，会移除未声明的包，因此临时工具优先使用 `uvx` 或
`uv run --with <package>`。

## 使用 Notebook

完整同步已经包含 `notebook` extra。启动项目内的 JupyterLab：

```bash
uv run --locked python -m ipykernel install --user --name scmm --display-name scMM
uv run --locked jupyter lab
```

打开项目根目录的 `scMM_workflow.ipynb`，选择显示名为 `scMM`、内部名称为 `scmm` 的内核。
VS Code 也可直接选择仓库内的 `.venv/bin/python`。Notebook 的功能和参数见
[参数化 Notebook 工作流](02-notebook-workflow.md)。

## 依赖分组

| 分组 | 主要用途 |
|---|---|
| 核心依赖 | 原始谱读取、矩阵处理、保存、归一化、基础统计 |
| `plot` | Matplotlib/Seaborn、UMAP、Palantir 和轨迹图 |
| `cluster` | Leiden 与 Louvain 聚类 |
| `ui` | Panel 引导式网页、Plotly 原始谱/质量图和 UMAP 质量检查 |
| `notebook` | JupyterLab 与项目内核 |
| `dev` 依赖组 | pytest、Ruff、构建与覆盖率；不进入发布包依赖 |

仅运行核心处理可使用：

```bash
uv sync --locked --no-dev
```

按需启用功能，例如：

```bash
uv sync --locked --extra plot --extra cluster
uv sync --locked --extra ui
```

仓库开发和 PR 验证统一使用 `uv sync --locked --all-extras --dev`，确保 notebook、分析和网页
入口共享同一份锁文件。

## 验证安装

```bash
uv run --locked python -c "import scMM, pyopenms, anndata; print('scMM environment ready')"
uv run --locked scmm-process --help
uv run --locked scmm-ui --help
uv run --locked pytest -q
```

如果入口不存在，先运行 `uv sync --locked --all-extras --dev`，并确认命令从包含 `pyproject.toml`
的仓库目录执行。

## 系统资源建议

原始目录处理会构建公共 m/z 网格，并可能并行读取多个谱文件。建议：

- 先用单个文件和 `N_JOBS=1` 验证参数。
- 再把 `N_JOBS` 增大；`-1` 会使用所有可用 CPU。
- 内存不足时减少并发，而不是优先降低质量分辨率。
- 将结果写到本地高速磁盘，完成后再归档到网络存储。

下一步：[Notebook 工作流](02-notebook-workflow.md)。
