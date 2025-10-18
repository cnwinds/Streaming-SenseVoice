#!/bin/bash
# 实时语音识别启动脚本

echo "启动SenseVoice实时语音识别服务..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3"
    exit 1
fi

# 检查依赖
echo "检查依赖包..."
python3 -c "import torch, funasr, gradio, fastapi, websockets" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "安装依赖包..."
    pip install -r requirements.txt
fi

# 设置环境变量
export SENSEVOICE_DEVICE=${SENSEVOICE_DEVICE:-cuda:0}

echo "设备: $SENSEVOICE_DEVICE"

# 选择启动模式
echo "请选择启动模式:"
echo "1) API服务 (端口50000)"
echo "2) 实时识别演示 (端口7860)"
echo "3) 实时识别API (端口8000)"
echo "4) 运行测试"
read -p "请输入选择 [1-4]: " choice

case $choice in
    1)
        echo "启动API服务..."
        python3 api.py
        ;;
    2)
        echo "启动实时识别演示..."
        python3 realtime_demo.py
        ;;
    3)
        echo "启动实时识别API..."
        python3 realtime_api.py
        ;;
    4)
        echo "运行测试..."
        python3 test_realtime.py
        ;;
    *)
        echo "无效选择"
        exit 1
        ;;
esac