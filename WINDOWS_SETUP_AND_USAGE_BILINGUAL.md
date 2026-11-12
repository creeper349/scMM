# SCMM Windows 配置与使用指南 / Windows Setup and Usage Guide

本文档适用于 64 位 Windows 10/11。SCMM 使用项目目录中的 `.scmm-env`（Conda 或 Python venv），网页界面默认运行在 `http://127.0.0.1:8501`。

This guide applies to 64-bit Windows 10/11. SCMM uses `.scmm-env` (Conda or Python venv) inside the project directory. The Web GUI runs at `http://127.0.0.1:8501` by default.

---

## 中文说明

### 1. 安装 Miniconda

1. 从 [Miniconda 官方页面](https://docs.conda.io/projects/miniconda/en/latest/) 下载 64 位 Windows 安装程序。
2. 完成安装后打开 **Miniconda Prompt**。
3. 如果 Conda 提示尚未接受 Anaconda 软件源协议，依次运行：

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

4. 关闭 Miniconda Prompt，并重新打开终端，使 Conda 命令生效。

已安装 64 位 Python 3.11 的电脑可以直接运行 `setup_scmm.cmd`。Python 和 Conda 均缺失时，脚本会显示安装说明并保留终端，完成手动安装后再次运行即可。

### 2. 自动配置 SCMM

将 SCMM 完整目录放在本地磁盘中。在资源管理器中双击：

```text
setup_scmm.cmd
```

也可以在 PowerShell 中运行：

```powershell
cd D:\SCMM\SCMM_CODE
.\setup_scmm.cmd
```

配置脚本将执行以下操作：

1. 复用已有环境，或查找已安装的 Conda / Python 3.11。
2. 创建或复用项目目录中的 `.scmm-env`。
3. 根据 `pyproject.toml` 检查 SCMM 和运行所需的 Python 包。
4. 将所有缺失包合并安装，已安装项目直接跳过。
5. 询问是否安装 Thermo RAW 直接读取组件。

脚本可以重复运行。更新 SCMM 源码或补充依赖时，再次运行脚本即可。

所有安装逻辑集中在 `setup_scmm.cmd`。只检查环境可运行 `setup_scmm.cmd --check`；指定 Python 可运行 `setup_scmm.cmd --python "C:\Python311\python.exe"`。安装与启动出错时均保留具体错误和退出码，按键后关闭终端。

### 3. Thermo RAW 直接读取组件

配置过程中将看到以下询问：

```text
Install this component and accept the license above? [y/N]
```

- 输入 `y`：安装 `pythonnet` 和 Thermo 官方 RawFileReader 程序集，支持逐帧直接读取 Thermo `.RAW` 文件。
- 输入 `n` 或直接按 Enter：跳过该组件，mzML 和 mzXML 功能仍可使用。

Thermo 组件安装到 `.scmm-env\Library\bin\ThermoRawFileReader`。组件完整存在时，后续配置会自动跳过。SCMM 的 Thermo RAW 读取过程直接调用官方接口，无需 MSConvert 和中间 mzML 文件。

### 4. 启动网页界面

配置完成后双击：

```text
start_scmm.cmd
```

或在 PowerShell 中运行：

```powershell
cd D:\SCMM\SCMM_CODE
.\start_scmm.cmd
```

浏览器将打开：

```text
http://127.0.0.1:8501
```

关闭启动终端或在终端按 `Ctrl+C` 可以停止 SCMM。

### 5. 网页端基本使用流程

1. **数据导入**：选择单个质谱文件或包含多个文件的目录。支持 mzML、mzXML，以及安装可选组件后的 Thermo RAW。
2. **处理参数**：设置 reference m/z、ppm 容差、m/z 范围、分辨率、细胞信号阈值和特征信噪比等参数。
3. **目录特征轴**：
   - `shared`：所有文件共同构建统一特征轴。
   - `independent`：每个文件独立提取特征，再按照 ppm 合并。
4. **运行处理**：任务终端显示当前主步骤及小步骤进度。
5. **检查结果**：在交互表格中查看 `X`、`obs` 和 `var`；细胞帧序号保存在 `obs["frame_id"]`。
6. **信号预览**：查看 TIC、EIC、MS1 质谱、叠加质谱和标记细胞帧的 EIC。
7. **后续分析**：执行归一化、插补、质控、降维、聚类、轨迹分析、监督学习和差异分析。
8. **结果导出**：保存 H5AD、CSV、质控报告和图形结果。

reference m/z 必须能在指定 ppm 容差内匹配到实际峰，否则处理将停止并报告缺失的参考峰。分析主流程只使用 MS1 数据，多级质谱扫描保留其级别信息并与 MS1 分开。

### 6. 命令行使用（可选）

处理单个文件：

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\sample.mzML" `
  "D:\results\sample.h5ad" `
  --reference-mz 734.5929 760.6085 `
  --ppm 10
```

处理 Thermo RAW 文件：

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\sample.RAW" `
  "D:\results\sample.h5ad" `
  --reference-mz 760.5847 `
  --ppm 10
```

处理目录并使用统一特征轴：

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\experiment" `
  "D:\results\experiment.h5ad" `
  --reference-mz 734.5929 760.6085 `
  --feature-strategy shared
```

检查和导出结果：

```powershell
.\.scmm-env\Scripts\scmm.exe inspect "D:\results\experiment.h5ad"
.\.scmm-env\Scripts\scmm.exe qc "D:\results\experiment.h5ad" "D:\results\qc.pdf"
.\.scmm-env\Scripts\scmm.exe export "D:\results\experiment.h5ad" "D:\results\csv"
```

查看完整参数：

```powershell
.\.scmm-env\Scripts\scmm.exe process --help
```

### 7. 更新和故障处理

- **更新代码后**：再次运行 `setup_scmm.cmd`。脚本会复用环境并补充缺失包。
- **需要稍后安装 RAW 组件**：再次运行 `setup_scmm.cmd`，在询问处输入 `y`。
- **8501 端口被占用**：在项目目录运行：

```powershell
.\.scmm-env\python.exe -m streamlit run .\scMM\webapp.py --server.port 8502
```

- **Conda 协议错误**：在 Miniconda Prompt 中执行第 1 节的三条 `conda tos accept` 命令。
- **单位电脑启用了应用控制策略**：请管理员将 Miniconda、项目目录和 `.scmm-env` 中的 Python 与二进制扩展加入允许列表。
- **环境需要重新创建**：先将 `.scmm-env` 改名作为备份，再运行 `setup_scmm.cmd` 创建新环境。确认新环境可用后再处理备份目录。

---

## English Instructions

### 1. Install Miniconda

1. Download the 64-bit Windows installer from the [official Miniconda page](https://docs.conda.io/projects/miniconda/en/latest/).
2. Complete the installation and open **Miniconda Prompt**.
3. If Conda reports that the Anaconda channel Terms of Service have not been accepted, run:

```powershell
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2
```

4. Close and reopen the terminal so that the Conda command is available.

Computers with 64-bit Python 3.11 can run `setup_scmm.cmd` directly. If both Python and Conda are missing, the installer displays installation instructions and keeps the terminal open. Run it again after completing the prerequisite installation.

### 2. Configure SCMM Automatically

Keep the complete SCMM directory on a local drive. Double-click:

```text
setup_scmm.cmd
```

Alternatively, run it from PowerShell:

```powershell
cd D:\SCMM\SCMM_CODE
.\setup_scmm.cmd
```

The setup script performs the following operations:

1. Reuses the existing environment, or finds Conda / Python 3.11.
2. Creates or reuses `.scmm-env` in the project directory.
3. Reads `pyproject.toml` and checks SCMM and its runtime Python distributions.
4. Installs the missing packages in one operation and skips installed packages.
5. Asks whether to install the Thermo RAW direct-reader component.

The script is safe to run again after updating SCMM or when additional dependencies are required.

All installation logic is contained in `setup_scmm.cmd`. Use `setup_scmm.cmd --check` for a read-only check, or `setup_scmm.cmd --python "C:\Python311\python.exe"` to select Python. Setup and startup failures display the error and exit code, and keep the terminal open until a key is pressed.

### 3. Thermo RAW Direct Reader

During setup, the following prompt appears:

```text
Install this component and accept the license above? [y/N]
```

- Enter `y` to install `pythonnet` and the official Thermo RawFileReader assemblies for direct frame-by-frame access to Thermo `.RAW` files.
- Enter `n`, or press Enter, to skip this optional component. mzML and mzXML processing remains available.

The Thermo assemblies are stored in `.scmm-env\Library\bin\ThermoRawFileReader`. A complete existing installation is detected and skipped on later setup runs. Thermo RAW files are read through the official API without MSConvert or an intermediate mzML file.

### 4. Start the Web GUI

After setup, double-click:

```text
start_scmm.cmd
```

Or run:

```powershell
cd D:\SCMM\SCMM_CODE
.\start_scmm.cmd
```

The browser opens:

```text
http://127.0.0.1:8501
```

Close the launcher terminal or press `Ctrl+C` in that terminal to stop SCMM.

### 5. Basic Web Workflow

1. **Import data**: select one MS file or a directory. Supported inputs are mzML, mzXML, and Thermo RAW when the optional reader is installed.
2. **Set processing parameters**: configure reference m/z, ppm tolerance, m/z range, resolution, cell-signal threshold, feature S/N threshold, and related parameters.
3. **Choose a directory feature-axis strategy**:
   - `shared`: build one common feature axis from all files.
   - `independent`: extract each file independently and merge features by ppm.
4. **Run processing**: the task terminal displays the current main-stage and sub-step progress.
5. **Inspect results**: view `X`, `obs`, and `var` in interactive tables. Cell-frame numbers are stored in `obs["frame_id"]`.
6. **Preview raw signals**: inspect TIC, EIC, MS1 spectra, summed spectra, and EIC traces with detected cell frames.
7. **Continue analysis**: run normalization, imputation, QC, dimension reduction, clustering, trajectory analysis, supervised learning, and differential analysis.
8. **Export results**: save H5AD, CSV, QC reports, and figures.

Each reference m/z must match an observed peak within the configured ppm tolerance. A missing reference peak stops processing and is reported. The main analysis pipeline uses MS1 scans; higher-level scans retain their level metadata and remain separated from MS1.

### 6. Command-Line Usage (Optional)

Process one file:

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\sample.mzML" `
  "D:\results\sample.h5ad" `
  --reference-mz 734.5929 760.6085 `
  --ppm 10
```

Process a Thermo RAW file:

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\sample.RAW" `
  "D:\results\sample.h5ad" `
  --reference-mz 760.5847 `
  --ppm 10
```

Process a directory with a shared feature axis:

```powershell
.\.scmm-env\Scripts\scmm.exe process `
  "D:\data\experiment" `
  "D:\results\experiment.h5ad" `
  --reference-mz 734.5929 760.6085 `
  --feature-strategy shared
```

Inspect and export results:

```powershell
.\.scmm-env\Scripts\scmm.exe inspect "D:\results\experiment.h5ad"
.\.scmm-env\Scripts\scmm.exe qc "D:\results\experiment.h5ad" "D:\results\qc.pdf"
.\.scmm-env\Scripts\scmm.exe export "D:\results\experiment.h5ad" "D:\results\csv"
```

Display all processing options:

```powershell
.\.scmm-env\Scripts\scmm.exe process --help
```

### 7. Updates and Troubleshooting

- **After updating the source**: run `setup_scmm.cmd` again. It reuses the environment and installs only missing packages.
- **To add RAW support later**: run `setup_scmm.cmd` again and enter `y` at the Thermo prompt.
- **If port 8501 is occupied**, run:

```powershell
.\.scmm-env\python.exe -m streamlit run .\scMM\webapp.py --server.port 8502
```

- **For a Conda Terms-of-Service error**: run the three `conda tos accept` commands from Section 1 in Miniconda Prompt.
- **On managed computers with application-control policies**: ask the administrator to allow Miniconda, the project directory, and Python/binary extensions inside `.scmm-env`.
- **To recreate the environment**: rename `.scmm-env` as a backup, then run `setup_scmm.cmd`. Remove the backup after confirming that the new environment works.
