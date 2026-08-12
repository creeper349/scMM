# 安装与环境配置

[返回文档索引](README.md)

## 推荐方式：Conda

PyOpenMS 等科学计算包包含编译组件，项目使用 Conda 保持版本组合一致：

```bash
conda env create -f environment.yml
conda activate scmm-dev
python -m pip install --no-deps -e .
```

这里的 `--no-deps` 是有意的：依赖已经由 `environment.yml` 安装，避免 pip 再解析并替换
Conda 提供的二进制包。`-e` 表示可编辑安装，修改本地 `scMM/` 代码后无需重新安装。

环境文件更新后执行：

```bash
conda env update -f environment.yml --prune
python -m pip install --no-deps -e .
```

`--prune` 会移除不再出现在环境定义中的包；如环境中还安装了个人使用的软件，执行前应先确认。

## 使用 Notebook

如果当前 Jupyter/JupyterLab 环境尚未提供 notebook 界面和内核注册工具，可在项目环境中安装：

```bash
conda install -n scmm-dev -c conda-forge jupyterlab ipykernel
conda activate scmm-dev
python -m ipykernel install --user --name scmm-dev --display-name scMM
jupyter lab
```

打开项目根目录的 `scMM_workflow.ipynb`，选择显示名为 `scMM` 的内核。Notebook 的功能和
参数见[参数化 Notebook 工作流](02-notebook-workflow.md)。

如果使用其他环境启动 Jupyter，也可以只把 `scmm-dev` 注册成内核，不需要在两个环境中
重复安装所有科学计算依赖。

## 依赖分组

| 分组 | 主要用途 |
|---|---|
| 核心依赖 | 原始谱读取、矩阵处理、保存、归一化、基础统计 |
| `plot` | Matplotlib/Seaborn、UMAP、Palantir 和轨迹图 |
| `cluster` | Leiden 与 Louvain 聚类 |
| `ui` | Panel 引导式网页、Plotly 原始谱/质量图和 UMAP 质量检查 |
| `dev` | pytest、Ruff、构建与覆盖率 |

`environment.yml` 已包含项目开发和完整分析所需的科学计算依赖。仅通过 pip 安装时可使用：

```bash
python -m pip install -e ".[plot,cluster,ui,dev]"
```

但对于 PyOpenMS，仍优先推荐 Conda。

## 验证安装

```bash
python -c "import scMM, pyopenms, anndata; print('scMM environment ready')"
scmm-process --help
scmm-ui --help
pytest -q
```

如果 `scmm-process` 或 `scmm-ui` 不存在，通常表示尚未执行可编辑安装，或当前 shell 没有激活
`scmm-dev`。

## 系统资源建议

原始目录处理会构建公共 m/z 网格，并可能并行读取多个谱文件。建议：

- 先用单个文件和 `N_JOBS=1` 验证参数。
- 再把 `N_JOBS` 增大；`-1` 会使用所有可用 CPU。
- 内存不足时减少并发，而不是优先降低质量分辨率。
- 将结果写到本地高速磁盘，完成后再归档到网络存储。

下一步：[Notebook 工作流](02-notebook-workflow.md)。
