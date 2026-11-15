# 更新到 SCMM-GUI-0.3.0-fix / Upgrade guide

## 新安装

1. 安装 64 位 Python 3.11 或 Miniconda。
2. 完整解压 `SCMM-GUI-0.3.0-fix.zip`。
3. 双击 `setup_scmm.cmd` 配置环境。按提示选择是否安装 Thermo RAW 读取组件。
4. 双击 `start_scmm.cmd`，在浏览器访问 `http://127.0.0.1:8501`。

安装需要联网下载依赖。安装与启动遇到错误时，终端会显示错误并等待按键。Windows 安装细节见 `INSTALL_SCMM_GUI.md`。

## 从 0.2.0 或 0.3.0 升级

1. 关闭正在运行的 SCMM，退出启动终端。
2. 下载与已安装版本对应的补丁。
   - 0.2.0 使用 `SCMM-GUI-0.3.0-fix-patch-from-0.2.0.zip`。
   - 0.3.0 使用 `SCMM-GUI-0.3.0-fix-patch-from-0.3.0.zip`。
3. 将补丁完整解压到原软件目录。`apply_patch_0.2.0.cmd` 或 `apply_patch_0.3.0.cmd` 应与原来的 `start_scmm.cmd` 同级。
4. 双击对应的 `apply_patch_*.cmd`。程序自动备份旧源码与配置、安装本版完整程序，并补齐缺失或不兼容的依赖。
5. 显示更新完成后，运行原目录中的 `start_scmm.cmd`。

补丁包含与完整发行版相同的程序、默认配置、界面资源和启动脚本。环境继续使用原目录的 `.scmm-env`。已有分析数据和结果保留。

旧源码、旧语言包和原参数文件位于 `.scmm-update-backups/更新时间/`。升级采用新版默认配置；有自定义参数时，可将备份中的 `scMM/config/defaults.json` 与新版配置合并。扩展语言包也可从备份复制到新版 `scMM/locales`。

如依赖配置失败，程序文件已更新，可处理错误后重跑 `setup_scmm.cmd`，或再次运行补丁。错误窗口中会保留具体原因。两类补丁均支持重复运行。

裁剪设置在“数据导入”页选择输入文件后配置。加载各文件裁剪滑块，拖动端点选择保留区间，然后开始峰提取。语言在左侧 `Language / 语言` 中选择。

## English

For a new installation, extract the full release, install 64-bit Python 3.11 or Miniconda, run `setup_scmm.cmd`, then run `start_scmm.cmd`.

For an existing installation, close SCMM and extract the matching patch into the original software folder. Keep the `apply_patch_*.cmd` launcher beside `start_scmm.cmd`, then run it. The updater backs up the old source/settings, installs the same program files as the full release, and installs missing or incompatible dependencies. Internet access is needed for dependency setup. Optional Thermo RAW support remains available through the installer license prompt.

Existing environments and analysis results are retained. The update uses the new default configuration. Previous custom settings and language packs are in `.scmm-update-backups` for merging. If environment setup fails, resolve the displayed error and rerun `setup_scmm.cmd` or the patch.
