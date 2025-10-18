#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# 实时语音识别演示应用

import gradio as gr
import numpy as np
import threading
import time
import json
from typing import List, Dict, Any, Optional
import logging

from realtime_asr import RealtimeASR, RecognitionResult
from realtime_api import RecognitionResponse

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealtimeASRDemo:
    """实时语音识别演示类"""
    
    def __init__(self):
        self.asr = None
        self.is_recording = False
        self.results = []
        self.current_speakers = set()
        
    def init_asr(self, model_dir: str, device: str, confidence_threshold: float, enable_speaker_id: bool):
        """初始化ASR"""
        try:
            self.asr = RealtimeASR(
                model_dir=model_dir,
                device=device,
                confidence_threshold=confidence_threshold,
                enable_speaker_id=enable_speaker_id
            )
            return "ASR初始化成功"
        except Exception as e:
            return f"ASR初始化失败: {str(e)}"
    
    def on_result_callback(self, result: RecognitionResult):
        """结果回调函数"""
        self.results.append(result)
        if result.speaker_id:
            self.current_speakers.add(result.speaker_id)
    
    def start_recording(self):
        """开始录音"""
        if not self.asr:
            return "请先初始化ASR", self.get_results_display()
        
        if self.is_recording:
            return "已经在录音中", self.get_results_display()
        
        try:
            self.asr.start_recording(callback=self.on_result_callback)
            self.is_recording = True
            return "开始录音成功", self.get_results_display()
        except Exception as e:
            return f"开始录音失败: {str(e)}", self.get_results_display()
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return "没有在录音", self.get_results_display()
        
        try:
            self.asr.stop_recording()
            self.is_recording = False
            return "停止录音成功", self.get_results_display()
        except Exception as e:
            return f"停止录音失败: {str(e)}", self.get_results_display()
    
    def clear_results(self):
        """清空结果"""
        self.results = []
        self.current_speakers = set()
        return "结果已清空", self.get_results_display()
    
    def get_results_display(self) -> str:
        """获取结果显示"""
        if not self.results:
            return "暂无识别结果"
        
        display_text = ""
        for i, result in enumerate(self.results[-20:], 1):  # 显示最近20条结果
            speaker_info = f"[{result.speaker_id}] " if result.speaker_id else ""
            confidence_info = f"({result.confidence:.2f})"
            timestamp_info = f"[{time.strftime('%H:%M:%S', time.localtime(result.timestamp))}]"
            
            display_text += f"{i}. {timestamp_info} {speaker_info}{confidence_info} {result.text}\n"
        
        return display_text
    
    def get_speakers_info(self) -> str:
        """获取说话人信息"""
        if not self.current_speakers:
            return "暂无说话人信息"
        
        speakers_text = "检测到的说话人:\n"
        for speaker in sorted(self.current_speakers):
            speakers_text += f"- {speaker}\n"
        
        return speakers_text
    
    def get_statistics(self) -> str:
        """获取统计信息"""
        if not self.results:
            return "暂无统计数据"
        
        total_results = len(self.results)
        avg_confidence = np.mean([r.confidence for r in self.results])
        total_speakers = len(self.current_speakers)
        
        # 按说话人统计
        speaker_counts = {}
        for result in self.results:
            if result.speaker_id:
                speaker_counts[result.speaker_id] = speaker_counts.get(result.speaker_id, 0) + 1
        
        stats_text = f"总识别次数: {total_results}\n"
        stats_text += f"平均置信度: {avg_confidence:.3f}\n"
        stats_text += f"说话人数量: {total_speakers}\n\n"
        
        if speaker_counts:
            stats_text += "各说话人识别次数:\n"
            for speaker, count in sorted(speaker_counts.items()):
                stats_text += f"- {speaker}: {count}次\n"
        
        return stats_text

# 创建演示实例
demo_app = RealtimeASRDemo()

def create_interface():
    """创建Gradio界面"""
    
    with gr.Blocks(title="实时语音识别演示", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🎤 实时语音识别系统")
        gr.Markdown("支持实时语音识别、置信度显示和声纹识别功能")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 配置")
                
                model_dir = gr.Textbox(
                    label="模型路径",
                    value="iic/SenseVoiceSmall",
                    placeholder="输入模型路径"
                )
                
                device = gr.Dropdown(
                    label="设备",
                    choices=["cuda:0", "cuda:1", "cpu"],
                    value="cuda:0"
                )
                
                confidence_threshold = gr.Slider(
                    label="置信度阈值",
                    minimum=0.1,
                    maximum=1.0,
                    value=0.6,
                    step=0.1
                )
                
                enable_speaker_id = gr.Checkbox(
                    label="启用声纹识别",
                    value=True
                )
                
                init_btn = gr.Button("初始化ASR", variant="primary")
                init_status = gr.Textbox(label="初始化状态", interactive=False)
                
            with gr.Column(scale=2):
                gr.Markdown("## 控制")
                
                with gr.Row():
                    start_btn = gr.Button("开始录音", variant="primary")
                    stop_btn = gr.Button("停止录音", variant="stop")
                    clear_btn = gr.Button("清空结果")
                
                control_status = gr.Textbox(label="控制状态", interactive=False)
                
                gr.Markdown("## 识别结果")
                results_display = gr.Textbox(
                    label="实时识别结果",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("## 说话人信息")
                speakers_info = gr.Textbox(
                    label="检测到的说话人",
                    lines=5,
                    interactive=False
                )
            
            with gr.Column():
                gr.Markdown("## 统计信息")
                statistics = gr.Textbox(
                    label="统计信息",
                    lines=5,
                    interactive=False
                )
        
        # 事件处理
        def init_asr_wrapper():
            status = demo_app.init_asr(
                model_dir.value,
                device.value,
                confidence_threshold.value,
                enable_speaker_id.value
            )
            return status, "", "", ""
        
        def start_recording_wrapper():
            status, results = demo_app.start_recording()
            speakers = demo_app.get_speakers_info()
            stats = demo_app.get_statistics()
            return status, results, speakers, stats
        
        def stop_recording_wrapper():
            status, results = demo_app.stop_recording()
            speakers = demo_app.get_speakers_info()
            stats = demo_app.get_statistics()
            return status, results, speakers, stats
        
        def clear_results_wrapper():
            status, results = demo_app.clear_results()
            speakers = demo_app.get_speakers_info()
            stats = demo_app.get_statistics()
            return status, results, speakers, stats
        
        def update_display():
            results = demo_app.get_results_display()
            speakers = demo_app.get_speakers_info()
            stats = demo_app.get_statistics()
            return results, speakers, stats
        
        # 绑定事件
        init_btn.click(
            init_asr_wrapper,
            outputs=[init_status, results_display, speakers_info, statistics]
        )
        
        start_btn.click(
            start_recording_wrapper,
            outputs=[control_status, results_display, speakers_info, statistics]
        )
        
        stop_btn.click(
            stop_recording_wrapper,
            outputs=[control_status, results_display, speakers_info, statistics]
        )
        
        clear_btn.click(
            clear_results_wrapper,
            outputs=[control_status, results_display, speakers_info, statistics]
        )
        
        # 自动更新显示
        interface.load(
            update_display,
            outputs=[results_display, speakers_info, statistics],
            every=1
        )
    
    return interface

def main():
    """主函数"""
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )

if __name__ == "__main__":
    main()