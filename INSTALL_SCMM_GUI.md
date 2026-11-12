# SCMM-GUI 安装与启动说明（Windows）

本发行包是 Python 源码版，不含、也不编译独立 EXE。安装完成后通过本机浏览器使用 SCMM-GUI。

## 1. 系统要求

- 64 位 Windows 10 或 Windows 11；
- 建议至少 8 GB 内存，大批量质谱数据建议 16 GB 或更多；
- 安装阶段需要连接互联网下载 Python 依赖；
- Edge、Chrome 或 Firefox 等现代浏览器。

## 2. 推荐安装方式：手动安装 Python

### 第一步：安装 Python 3.11

从 Python 官方网站下载并安装 **64 位 Python 3.11**。安装界面务必勾选：

```text
Add python.exe to PATH
```

安装完成后打开新的 PowerShell 或 CMD，执行：

```powershell
python --version
```

应显示 `Python 3.11.x`。不要使用 Microsoft Store 的 Python 占位程序；若输入 `python` 跳转商店，请在 Windows“管理应用执行别名”中关闭 `python.exe` 和 `python3.exe` 别名，然后重新安装官方 Python。

### 第二步：解压发行包

将 ZIP 完整解压到普通目录，例如：

```text
D:\SCMM-GUI
```

不要直接在 ZIP 预览窗口中运行脚本，也不要放在需要管理员权限的 `Program Files` 目录。

### 第三步：自动配置虚拟环境

双击：

```text
setup_scmm.cmd
```

脚本会在发行目录内创建 `.scmm-env`，并安装 SCMM、Streamlit、pyOpenMS、scikit-learn、SHAP、Leiden/Louvain 等依赖。根据网速通常需要数分钟。安装不需要管理员权限。

如双击失败，可在该目录打开 PowerShell，执行：

```powershell
.\setup_scmm.cmd
```

若 Python 命令不是 `python`，可明确指定路径：

```powershell
.\setup_scmm.cmd --python "C:\Python311\python.exe"
```

### 第四步：启动

双击：

```text
start_scmm.cmd
```

保持终端窗口开启。默认浏览器会打开 `http://127.0.0.1:8501`。使用结束后关闭浏览器页面，再在终端按 `Ctrl+C` 停止服务。

## 3. 使用已安装的 Miniconda

已安装 Miniconda 时，双击 `setup_scmm.cmd` 即可创建或复用 `.scmm-env`。缺少 Python/Conda 时，脚本显示手动安装说明。安装和启动失败时保留具体错误与退出码，等待按键后关闭。

配置逻辑全部位于 `setup_scmm.cmd`，同时兼容 Conda 和 Python venv。已有 `.ps1` 和 `install_with_python.cmd` 入口转发到这一批处理文件。`setup_scmm.cmd --check` 可只检查现有环境。

## 4. 需要另外手动安装的可选内容

### 处理 mzML、mzXML 或 H5AD

除 Python/Conda 环境外，不需要其它软件。

### 直接处理 Thermo RAW 文件

运行 `setup_scmm.cmd`，在 Thermo 许可询问处输入 `y`。脚本按需安装 pythonnet 和官方 RawFileReader 组件，随后可以逐帧直接读取 Thermo RAW。

## 5. 参数配置

默认参数与参数帮助文字保存在：

```text
scMM\config\defaults.json
```

关闭 SCMM-GUI 后用 UTF-8 文本编辑器修改：

- `default`：控件初始值；
- `help`：光标悬停参数名旁帮助图标时显示的说明。

必须保持 JSON 语法合法，字符串用双引号，布尔值用 `true` 或 `false`。

## 6. 更新、重装和卸载

- 更新源码后，再运行一次 `install_with_python.cmd` 或 `setup_scmm.cmd` 更新依赖；
- 环境损坏时，关闭 SCMM，删除发行目录中的 `.scmm-env`，再重新安装；
- 卸载时直接删除整个 SCMM-GUI 目录；
- 单独安装的 Miniconda/Python 可继续供其它项目使用。

### 已经安装过 SCMM 的电脑如何使用新版本

1. 关闭旧版 SCMM-GUI，并备份旧目录中的 `results` 和修改过的 `scMM\config\defaults.json`。
2. 解压新版本，把新版本文件覆盖到原 SCMM-GUI 目录；保留原目录中的 `.scmm-env`，无需重新下载整个 Python 环境。
3. 如果原来通过 `install_with_python.cmd` 安装，重新运行一次该脚本；如果原来通过 `setup_scmm.cmd`/Miniconda 安装，重新运行一次 `setup_scmm.cmd`。脚本会把现有环境中的 SCMM 指向新源码并补齐新增依赖。
4. 双击 `start_scmm.cmd` 启动新版。

`.scmm-env\Scripts\python.exe` 表示 Python venv，`.scmm-env\python.exe` 表示 Conda 环境；两者都使用 `setup_scmm.cmd` 配置。换电脑时，在新目录运行配置脚本创建环境即可。

## 7. 常见问题

### Windows 应用控制或杀毒软件拦截 Python/DLL

将发行目录和官方 Python 安装目录放到允许列表；不要从聊天软件临时目录或网络共享路径运行。若单位电脑由管理员实施应用控制策略，需要管理员允许官方 Python 及 `.scmm-env` 中的二进制扩展，普通用户无法安全绕过该策略。

### 浏览器没有自动打开

保持启动终端开启，手动访问 `http://127.0.0.1:8501`。服务只监听本机地址，不会向局域网开放。

### 8501 端口已被占用

先关闭旧的 SCMM/Streamlit 终端；或执行：

```powershell
.\.scmm-env\Scripts\python.exe -m streamlit run .\scMM\webapp.py --server.port 8502
```

若使用 Miniconda 环境，则 Python 路径为 `.\.scmm-env\python.exe`。

### 安装中断

确认网络可访问 PyPI/Conda，重新运行安装脚本即可。若仍失败，保留终端中的完整错误信息用于排查。
