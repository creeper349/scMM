@echo off
setlocal
call "%~dp0setup_scmm.cmd" --python python %*
exit /b %errorlevel%
