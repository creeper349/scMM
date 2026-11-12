@echo off
setlocal EnableExtensions DisableDelayedExpansion
chcp 65001 >nul
title SCMM Web GUI
set "SCMM_START_EXIT="
pushd "%~dp0"
if errorlevel 1 goto failed
if not exist "scMM\webapp.py" (
  echo [SCMM] ERROR: scMM\webapp.py is missing. Keep the complete project next to this launcher.
  goto failed
)
call :find_python
if not defined SCMM_PYTHON (
  echo [SCMM] The local environment is missing. Running setup...
  call "%~dp0setup_scmm.cmd" --no-pause
  if errorlevel 1 goto failed
  call :find_python
)
if not defined SCMM_PYTHON (
  echo [SCMM] ERROR: Python was not found in .scmm-env after setup.
  goto failed
)
echo [SCMM] Python: %SCMM_PYTHON%
echo [SCMM] Web address: http://127.0.0.1:8501
echo [SCMM] Keep this window open. Press Ctrl+C to stop SCMM.
"%SCMM_PYTHON%" -m streamlit run "scMM\webapp.py" --server.address=127.0.0.1 --server.port=8501 --server.headless=false --browser.gatherUsageStats=false
set "SCMM_START_EXIT=%errorlevel%"
if not "%SCMM_START_EXIT%"=="0" goto failed
popd
exit /b 0

:find_python
set "SCMM_PYTHON="
if exist ".scmm-env\Scripts\python.exe" set "SCMM_PYTHON=%CD%\.scmm-env\Scripts\python.exe"
if exist ".scmm-env\python.exe" set "SCMM_PYTHON=%CD%\.scmm-env\python.exe"
exit /b 0

:failed
if not defined SCMM_START_EXIT set "SCMM_START_EXIT=1"
echo.
echo [SCMM] ERROR: Startup failed with exit code %SCMM_START_EXIT%.
echo [SCMM] Review the error above. Run setup_scmm.cmd to check or complete the environment.
echo [SCMM] The window stays open until you press a key.
pause
exit /b %SCMM_START_EXIT%
