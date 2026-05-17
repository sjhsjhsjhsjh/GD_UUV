#!/bin/bash
# Linux/Mac 启动脚本：强化学习轨迹回放面板服务
#
# 功能说明：
#   1. 设置 Python 可执行路径
#   2. 检查依赖（Flask, Flask-CORS）
#   3. 启动 Flask 服务
#   4. 尝试自动打开浏览器

set -e

echo "========================================================================"
echo " 强化学习轨迹回放面板 - 服务启动脚本"
echo "========================================================================"
echo ""

# 设置 Python 可执行路径
# 根据操作系统调整路径
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PYTHON_EXE="/opt/miniconda/envs/torch_gpu/bin/python"  # Linux 示例
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHON_EXE="/opt/miniconda/envs/torch_gpu/bin/python"  # macOS 示例
else
    PYTHON_EXE="python"  # 回退方案
fi

# 尝试使用项目中指定的 Python 可执行文件（如果存在）
if [ -f "E:/lib/conda-env/torch_gpu/bin/python" ]; then
    PYTHON_EXE="E:/lib/conda-env/torch_gpu/bin/python"
fi

# 检查 Python 是否存在
if ! command -v "$PYTHON_EXE" &> /dev/null; then
    echo "[ERROR] Python 可执行文件不存在: $PYTHON_EXE"
    echo "请检查 Python 环境配置"
    exit 1
fi

echo "[INFO] Python 可执行文件: $PYTHON_EXE"
echo ""

# 获取当前脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "[INFO] 当前工作目录: $(pwd)"
echo ""

# 检查依赖
echo "[INFO] 检查依赖..."

"$PYTHON_EXE" -c "import flask" 2>/dev/null || {
    echo "[WARN] Flask 未安装，尝试安装..."
    "$PYTHON_EXE" -m pip install flask flask-cors -q
}

"$PYTHON_EXE" -c "import flask_cors" 2>/dev/null || {
    echo "[WARN] Flask-CORS 未安装，尝试安装..."
    "$PYTHON_EXE" -m pip install flask-cors -q
}

echo "[SUCCESS] 依赖检查完成"
echo ""

# 启动 Flask 服务
echo "[INFO] 启动 Flask 服务..."
echo ""

sleep 1

# 启动 Flask
"$PYTHON_EXE" server.py &
FLASK_PID=$!

# 等待服务启动
sleep 3

# 尝试打开浏览器
echo "[INFO] 尝试打开浏览器..."

if command -v xdg-open &> /dev/null; then
    xdg-open http://127.0.0.1:5000/ 2>/dev/null || true
elif command -v open &> /dev/null; then
    open http://127.0.0.1:5000/ 2>/dev/null || true
elif command -v sensible-browser &> /dev/null; then
    sensible-browser http://127.0.0.1:5000/ 2>/dev/null || true
else
    echo "[WARN] 无法自动打开浏览器，请手动访问: http://127.0.0.1:5000/"
fi

echo ""
echo "========================================================================"
echo " 服务已启动！"
echo " 前端地址: http://127.0.0.1:5000/"
echo " 后端 API: http://127.0.0.1:5000/api/"
echo ""
echo " 按 Ctrl+C 停止服务"
echo "========================================================================"
echo ""

# 保持脚本运行，等待 Flask 进程结束
wait $FLASK_PID
