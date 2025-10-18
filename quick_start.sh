#!/bin/bash

# SenseVoice 实时语音识别快速启动脚本

echo "========================================"
echo "SenseVoice 实时语音识别系统"
echo "========================================"
echo ""

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装 Python 3.8+"
    exit 1
fi

echo "检测到 Python 版本:"
python3 --version
echo ""

# 显示菜单
echo "请选择操作:"
echo "1) 安装/更新依赖"
echo "2) 测试新功能"
echo "3) 启动实时识别 Web 服务"
echo "4) 运行命令行演示（需要音频文件）"
echo "5) 查看文档"
echo "6) 退出"
echo ""

read -p "请输入选项 (1-6): " choice

case $choice in
    1)
        echo ""
        echo "正在安装/更新依赖..."
        pip install -r requirements.txt
        echo ""
        echo "✓ 依赖安装完成"
        ;;
    2)
        echo ""
        echo "正在运行功能测试..."
        python3 test_new_features.py
        ;;
    3)
        echo ""
        echo "正在启动实时识别 Web 服务..."
        echo "服务将在 http://localhost:8000 运行"
        echo "按 Ctrl+C 停止服务"
        echo ""
        
        # 设置设备
        if command -v nvidia-smi &> /dev/null; then
            echo "检测到 NVIDIA GPU，使用 CUDA"
            export SENSEVOICE_DEVICE=cuda:0
        else
            echo "未检测到 GPU，使用 CPU"
            export SENSEVOICE_DEVICE=cpu
        fi
        
        python3 realtime_api.py
        ;;
    4)
        echo ""
        read -p "请输入音频文件路径: " audio_file
        
        if [ ! -f "$audio_file" ]; then
            echo "错误: 文件不存在: $audio_file"
            exit 1
        fi
        
        echo ""
        echo "选择模式:"
        echo "1) 简单模式"
        echo "2) 实时流式模式"
        read -p "请选择 (1-2): " mode_choice
        
        if [ "$mode_choice" = "2" ]; then
            mode="realtime"
        else
            mode="simple"
        fi
        
        echo ""
        echo "正在识别: $audio_file"
        python3 demo_realtime.py --audio_file "$audio_file" --mode "$mode"
        ;;
    5)
        echo ""
        echo "文档列表:"
        echo "- README_REALTIME.md: 详细使用文档"
        echo "- CHANGES_SUMMARY.md: 改动总结"
        echo "- README_zh.md: 原始中文文档"
        echo ""
        
        if command -v cat &> /dev/null; then
            read -p "是否查看 README_REALTIME.md? (y/n): " view_doc
            if [ "$view_doc" = "y" ] || [ "$view_doc" = "Y" ]; then
                cat README_REALTIME.md | less
            fi
        fi
        ;;
    6)
        echo "退出"
        exit 0
        ;;
    *)
        echo "无效选项"
        exit 1
        ;;
esac

echo ""
echo "操作完成！"
echo ""
echo "更多信息请查看:"
echo "- README_REALTIME.md"
echo "- python3 demo_realtime.py --help"
echo "- python3 realtime_api.py (启动 Web 服务)"
