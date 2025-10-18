#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
快速测试新功能脚本
用于验证实时识别、置信度输出和声纹识别功能
"""

import torch
from model import SenseVoiceSmall
from speaker_embedding import SpeakerIdentifier
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def test_basic_features():
    """测试基础功能"""
    print("=" * 80)
    print("测试 1: 基础识别功能（原有功能）")
    print("=" * 80)
    
    model_dir = "iic/SenseVoiceSmall"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"\n使用设备: {device}")
    print("正在加载模型...")
    
    m, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device=device)
    m.eval()
    
    print("✓ 模型加载成功\n")
    
    # 使用示例音频
    example_audio = f"{kwargs['model_path']}/example/en.mp3"
    
    print(f"正在识别: {example_audio}")
    res = m.inference(
        data_in=example_audio,
        language="auto",
        use_itn=True,
        **kwargs,
    )
    
    if res and res[0]:
        text = rich_transcription_postprocess(res[0][0]["text"])
        print(f"\n识别结果: {text}")
        print("\n✓ 基础功能测试通过")
    else:
        print("\n✗ 基础功能测试失败")
    
    return m, kwargs


def test_confidence_output(model, kwargs):
    """测试置信度输出"""
    print("\n" + "=" * 80)
    print("测试 2: Token 置信度输出（新功能）")
    print("=" * 80)
    
    example_audio = f"{kwargs['model_path']}/example/en.mp3"
    
    print(f"\n正在识别并输出置信度: {example_audio}")
    res = model.inference(
        data_in=example_audio,
        language="en",
        use_itn=True,
        output_confidence=True,  # 新增参数
        **kwargs,
    )
    
    if res and res[0]:
        result = res[0][0]
        text = rich_transcription_postprocess(result["text"])
        
        print(f"\n识别结果: {text}")
        
        if "average_confidence" in result:
            print(f"\n平均置信度: {result['average_confidence']*100:.2f}%")
        
        if "token_confidences" in result:
            print(f"\nToken 数量: {len(result['token_confidences'])}")
            print("\n前 5 个 Token 的置信度:")
            for i, tc in enumerate(result['token_confidences'][:5], 1):
                print(f"  {i}. '{tc['token']}' - {tc['confidence']*100:.2f}%")
            
            print("\n✓ 置信度输出功能测试通过")
        else:
            print("\n✗ 未找到置信度信息")
    else:
        print("\n✗ 置信度输出功能测试失败")


def test_speaker_identification(model, kwargs):
    """测试声纹识别"""
    print("\n" + "=" * 80)
    print("测试 3: 声纹识别和说话人唯一标识（新功能）")
    print("=" * 80)
    
    # 创建说话人识别器
    speaker_identifier = SpeakerIdentifier(
        similarity_threshold=0.75,
        embedding_dim=256
    )
    
    # 测试音频列表（使用同一个音频多次以模拟）
    example_audio = f"{kwargs['model_path']}/example/en.mp3"
    test_audios = [example_audio] * 2  # 使用相同音频两次，应识别为同一说话人
    
    print(f"\n测试说话人识别（使用 {len(test_audios)} 个音频）")
    
    for i, audio in enumerate(test_audios, 1):
        print(f"\n--- 音频 {i} ---")
        
        res = model.inference(
            data_in=audio,
            language="auto",
            use_itn=True,
            output_confidence=True,
            output_speaker_embedding=True,  # 新增参数
            speaker_identifier=speaker_identifier,  # 新增参数
            **kwargs,
        )
        
        if res and res[0]:
            result = res[0][0]
            text = rich_transcription_postprocess(result["text"])
            
            print(f"文本: {text}")
            
            if "speaker_info" in result:
                speaker_info = result["speaker_info"]
                print(f"说话人ID: {speaker_info['speaker_id']}")
                
                if speaker_info.get('similarity') is not None:
                    print(f"相似度: {speaker_info['similarity']*100:.2f}%")
                    print("状态: 已识别的说话人")
                else:
                    print("状态: 新说话人")
            
            if "speaker_embedding" in result:
                embedding_len = len(result["speaker_embedding"])
                print(f"声纹向量维度: {embedding_len}")
                print(f"声纹向量样本: [{result['speaker_embedding'][0]:.4f}, "
                      f"{result['speaker_embedding'][1]:.4f}, ...]")
    
    # 统计
    all_speakers = speaker_identifier.get_all_speakers()
    print(f"\n检测到的不同说话人数量: {len(all_speakers)}")
    print(f"说话人列表: {all_speakers}")
    
    if len(all_speakers) > 0:
        print("\n✓ 声纹识别功能测试通过")
    else:
        print("\n✗ 声纹识别功能测试失败")


def test_realtime_stream():
    """测试实时流处理模块"""
    print("\n" + "=" * 80)
    print("测试 4: 实时流处理模块（新功能）")
    print("=" * 80)
    
    try:
        from realtime_stream import AudioStreamBuffer, RealtimeStreamManager
        import numpy as np
        
        # 测试音频缓冲区
        print("\n测试音频缓冲区...")
        buffer = AudioStreamBuffer(sample_rate=16000, chunk_size=1600)
        
        # 添加测试数据
        test_audio = np.random.randn(3200).astype(np.float32)
        buffer.add_audio(test_audio[:1600])
        buffer.add_audio(test_audio[1600:])
        
        print(f"缓冲区大小: {buffer.size()}")
        
        # 获取数据
        retrieved_audio = buffer.get_audio(1600)
        print(f"获取的数据大小: {len(retrieved_audio)}")
        
        if len(retrieved_audio) == 1600:
            print("\n✓ 实时流处理模块测试通过")
        else:
            print("\n✗ 实时流处理模块测试失败")
            
    except Exception as e:
        print(f"\n✗ 实时流处理模块测试失败: {e}")


def test_all_features():
    """测试所有新功能"""
    print("\n" + "🎯" * 40)
    print("SenseVoice 增强功能测试")
    print("🎯" * 40 + "\n")
    
    try:
        # 测试 1: 基础功能
        model, kwargs = test_basic_features()
        
        # 测试 2: 置信度输出
        test_confidence_output(model, kwargs)
        
        # 测试 3: 声纹识别
        test_speaker_identification(model, kwargs)
        
        # 测试 4: 实时流处理
        test_realtime_stream()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试完成！")
        print("=" * 80)
        print("\n新增功能总结:")
        print("1. ✓ 实时流式语音识别")
        print("2. ✓ Token 级别置信度输出")
        print("3. ✓ 声纹识别和说话人唯一标识")
        print("4. ✓ WebSocket 实时 API（请运行 realtime_api.py 测试）")
        print("\n使用说明:")
        print("- 查看 README_REALTIME.md 了解详细文档")
        print("- 运行 'python realtime_api.py' 启动实时识别服务")
        print("- 运行 'python demo_realtime.py --help' 查看演示程序用法")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_all_features()
