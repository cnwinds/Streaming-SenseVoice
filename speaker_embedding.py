# -*- encoding: utf-8 -*-
"""
声纹识别模块
支持说话人嵌入提取和唯一标识生成
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import hashlib


class SpeakerEmbeddingExtractor(nn.Module):
    """说话人嵌入提取器
    
    基于encoder输出提取说话人特征向量
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        embedding_dim: int = 256,
        hidden_dim: int = 512
    ):
        """
        Args:
            input_dim: 输入特征维度
            embedding_dim: 嵌入向量维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # 统计池化层
        self.pooling = StatisticsPooling()
        
        # 说话人嵌入网络
        self.speaker_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
    def forward(self, encoder_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_out: 编码器输出 (batch, time, dim)
            lengths: 序列长度 (batch,)
            
        Returns:
            speaker_embedding: 说话人嵌入 (batch, embedding_dim)
        """
        # 统计池化
        pooled = self.pooling(encoder_out, lengths)
        
        # 提取说话人嵌入
        speaker_embedding = self.speaker_encoder(pooled)
        
        # L2归一化
        speaker_embedding = F.normalize(speaker_embedding, p=2, dim=1)
        
        return speaker_embedding


class StatisticsPooling(nn.Module):
    """统计池化层
    
    计算均值和标准差并拼接
    """
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征 (batch, time, dim)
            lengths: 序列长度 (batch,)
            
        Returns:
            pooled: 池化后的特征 (batch, dim*2)
        """
        batch_size = x.size(0)
        
        # 创建mask
        max_len = x.size(1)
        mask = torch.arange(max_len, device=x.device).expand(batch_size, max_len) < lengths.unsqueeze(1)
        mask = mask.unsqueeze(2)  # (batch, time, 1)
        
        # 计算均值
        masked_x = x * mask.float()
        mean = masked_x.sum(dim=1) / lengths.unsqueeze(1).float()
        
        # 计算标准差
        squared_diff = ((x - mean.unsqueeze(1)) ** 2) * mask.float()
        std = torch.sqrt(squared_diff.sum(dim=1) / lengths.unsqueeze(1).float() + 1e-8)
        
        # 拼接均值和标准差
        pooled = torch.cat([mean, std], dim=1)
        
        return pooled


class SpeakerIdentifier:
    """说话人识别器
    
    用于生成说话人唯一标识和相似度计算
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        embedding_dim: int = 256
    ):
        """
        Args:
            similarity_threshold: 相似度阈值，超过此值认为是同一说话人
            embedding_dim: 嵌入向量维度
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_dim = embedding_dim
        self.speaker_database = {}  # 存储已知说话人的嵌入
        self.speaker_counter = 0  # 说话人计数器
        
    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """计算余弦相似度"""
        # 确保是单位向量
        embedding1 = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        
        # 计算余弦相似度
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
        
    def identify_speaker(
        self,
        embedding: np.ndarray,
        return_similarity: bool = True
    ) -> Tuple[str, Optional[float]]:
        """识别或注册说话人
        
        Args:
            embedding: 说话人嵌入向量
            return_similarity: 是否返回相似度
            
        Returns:
            speaker_id: 说话人唯一标识
            similarity: 与数据库中最相似说话人的相似度（如果是新说话人则为None）
        """
        embedding = embedding.flatten()
        
        if len(self.speaker_database) == 0:
            # 第一个说话人
            speaker_id = self._generate_speaker_id(embedding)
            self.speaker_database[speaker_id] = embedding
            return speaker_id, None if return_similarity else speaker_id
            
        # 计算与所有已知说话人的相似度
        max_similarity = -1.0
        best_speaker_id = None
        
        for sid, stored_embedding in self.speaker_database.items():
            similarity = self.compute_similarity(embedding, stored_embedding)
            if similarity > max_similarity:
                max_similarity = similarity
                best_speaker_id = sid
                
        # 判断是否为已知说话人
        if max_similarity >= self.similarity_threshold:
            if return_similarity:
                return best_speaker_id, max_similarity
            return best_speaker_id
        else:
            # 新说话人
            speaker_id = self._generate_speaker_id(embedding)
            self.speaker_database[speaker_id] = embedding
            if return_similarity:
                return speaker_id, None
            return speaker_id
            
    def _generate_speaker_id(self, embedding: np.ndarray) -> str:
        """生成说话人唯一标识"""
        # 使用嵌入向量的哈希值生成ID
        embedding_bytes = embedding.tobytes()
        hash_obj = hashlib.sha256(embedding_bytes)
        speaker_hash = hash_obj.hexdigest()[:16]
        
        self.speaker_counter += 1
        speaker_id = f"speaker_{self.speaker_counter:04d}_{speaker_hash}"
        
        return speaker_id
        
    def get_all_speakers(self) -> list:
        """获取所有已知说话人ID"""
        return list(self.speaker_database.keys())
        
    def clear_database(self):
        """清空说话人数据库"""
        self.speaker_database.clear()
        self.speaker_counter = 0
        
    def save_database(self, filepath: str):
        """保存说话人数据库"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'database': self.speaker_database,
                'counter': self.speaker_counter
            }, f)
            
    def load_database(self, filepath: str):
        """加载说话人数据库"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.speaker_database = data['database']
            self.speaker_counter = data['counter']


def extract_speaker_embedding(
    model,
    encoder_out: torch.Tensor,
    encoder_out_lens: torch.Tensor,
    speaker_extractor: Optional[SpeakerEmbeddingExtractor] = None
) -> torch.Tensor:
    """从编码器输出提取说话人嵌入
    
    Args:
        model: SenseVoice模型
        encoder_out: 编码器输出
        encoder_out_lens: 编码器输出长度
        speaker_extractor: 说话人嵌入提取器（如果为None则使用简单池化）
        
    Returns:
        speaker_embedding: 说话人嵌入向量
    """
    if speaker_extractor is not None:
        # 使用专门的提取器
        return speaker_extractor(encoder_out, encoder_out_lens)
    else:
        # 使用简单的平均池化
        batch_size = encoder_out.size(0)
        speaker_embeddings = []
        
        for i in range(batch_size):
            # 获取有效长度的特征
            valid_len = encoder_out_lens[i].item()
            features = encoder_out[i, :valid_len, :]
            
            # 平均池化
            mean_features = features.mean(dim=0)
            
            # L2归一化
            embedding = F.normalize(mean_features.unsqueeze(0), p=2, dim=1)
            speaker_embeddings.append(embedding)
            
        return torch.cat(speaker_embeddings, dim=0)
