@echo off
rem =====================================================================
rem 父子分块 RAG 检索系统 —— Windows 双击「运行按钮」
rem 双击（无参数）＝ 自动重启（先停后起），全程无需任何输入，结束保留窗口便于查看结果
rem 终端其它命令： start.bat status / stop / restart / selftest / logs
rem =====================================================================
setlocal
chcp 65001 >nul
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo [错误] 未找到 python 命令，请先安装 Python 3.10+ 并加入 PATH。
  pause
  exit /b 1
)

python "%~dp0scripts\start.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" goto failed
rem 无参数（双击场景）时保留窗口，便于查看启动结果面板
if "%~1"=="" pause
exit /b 0

:failed
echo.
echo [失败] 退出码 %RC%，请按上面提示排查。
pause
exit /b %RC%