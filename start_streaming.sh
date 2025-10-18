#!/bin/bash
# 启动流式语音识别服务

echo "========================================"
echo "  SenseVoice 流式语音识别服务"
echo "========================================"
echo ""

# 检查设备
if [ -z "$SENSEVOICE_DEVICE" ]; then
    # 尝试检测GPU
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            export SENSEVOICE_DEVICE="cuda:0"
            echo "✓ 检测到GPU，使用设备: cuda:0"
        else
            export SENSEVOICE_DEVICE="cpu"
            echo "⚠ 未检测到可用GPU，使用CPU"
        fi
    else
        export SENSEVOICE_DEVICE="cpu"
        echo "⚠ nvidia-smi不可用，使用CPU"
    fi
else
    echo "✓ 使用指定设备: $SENSEVOICE_DEVICE"
fi

# 设置端口
if [ -z "$PORT" ]; then
    export PORT=8001
fi

echo "✓ 端口: $PORT"
echo ""

echo "正在启动服务..."
echo ""
echo "服务启动后，请访问:"
echo "  🌐 主页: http://localhost:$PORT"
echo "  🎬 流式识别演示: http://localhost:$PORT/streaming-demo"
echo "  📚 API文档: http://localhost:$PORT/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"
echo ""

# 启动服务
python streaming_api.py
