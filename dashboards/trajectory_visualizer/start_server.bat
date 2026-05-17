@echo off
REM Windows 启动脚本：强化学习轨迹回放面板服务
REM 
REM 功能说明：
REM   1. 设置 Python 可执行路径
REM   2. 检查依赖（Flask, Flask-CORS）
REM   3. 启动 Flask 服务
REM   4. 自动打开浏览器

chcp 65001 > nul
cls

echo ========================================================================
echo  强化学习轨迹回放面板 - 服务启动脚本
echo ========================================================================
echo.

REM 设置 Python 可执行路径
set PYTHON_EXE=E:\lib\conda-env\torch_gpu\python.exe

REM 检查 Python 是否存在
if not exist "%PYTHON_EXE%" (
    echo [ERROR] Python 可执行文件不存在: %PYTHON_EXE%
    echo 请检查 Python 环境配置
    pause
    exit /b 1
)

echo [INFO] Python 可执行文件: %PYTHON_EXE%
echo.

REM 获取当前脚本所在目录（项目根目录的相对位置）
for %%A in ("%~dp0") do set SCRIPT_DIR=%%~dpA
cd /d "%SCRIPT_DIR%"

echo [INFO] 当前工作目录: %cd%
echo.

REM 检查依赖
echo [INFO] 检查依赖...
"%PYTHON_EXE%" -c "import flask" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Flask 未安装，尝试安装...
    "%PYTHON_EXE%" -m pip install flask flask-cors -q
)

"%PYTHON_EXE%" -c "import flask_cors" >nul 2>&1
if errorlevel 1 (
    echo [WARN] Flask-CORS 未安装，尝试安装...
    "%PYTHON_EXE%" -m pip install flask-cors -q
)

echo [SUCCESS] 依赖检查完成
echo.

REM 启动 Flask 服务
echo [INFO] 启动 Flask 服务...
echo.

timeout /t 1 > nul

REM 在新窗口中启动 Flask，并保持窗口打开
start "Trajectory Visualizer Backend" cmd /k "^
echo. & ^
echo 服务正在启动... & ^
echo. & ^
cd /d "%SCRIPT_DIR%" & ^
"%PYTHON_EXE%" server.py & ^
echo. & ^
echo 服务已停止（关闭此窗口） & ^
pause"

REM 等待服务启动
echo [INFO] 等待服务启动...
timeout /t 3 > nul

REM 尝试打开浏览器
echo [INFO] 尝试打开浏览器...
start http://127.0.0.1:5000/

echo.
echo ========================================================================
echo  服务已启动！
echo  前端地址: http://127.0.0.1:5000/
echo  后端 API: http://127.0.0.1:5000/api/
echo ========================================================================
echo.
echo 按任意键关闭此窗口...
pause > nul
