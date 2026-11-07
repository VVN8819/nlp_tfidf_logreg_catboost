# bj_models_dl.py

"""
Архитектуры Deep Learning моделей для классификации текстов.
"""

import torch
import torch.nn as nn
from . import ba_config

class TextFCN(nn.Module):
    """
    Fully Connected Network (Многослойный перцептрон) для текстов.
    Архитектура: Embedding -> Mean Pooling (с маской) -> FC Layers -> Output
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_dim=256, dropout=0.5):
        super(TextFCN, self).__init__()
        
        # 1. Embedding слой
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=ba_config.PAD_IDX)
        
        # 2. Полносвязные слои
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
        # 3. Активация и Dropout
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: тензор индексов слов формы (batch_size, seq_length)
        """
        # Превращаем индексы в плотные векторы
        embedded = self.embedding(x) # (batch_size, seq_length, embedding_dim)
        
        # Mean Pooling с маской
        mask = (x != ba_config.PAD_IDX).unsqueeze(-1).float()
        
        # Обнуляем эмбеддинги паддинга
        masked_embedded = embedded * mask
        
        # Суммируем по длине последовательности и делим на реальное количество слов
        # clamp(min=1) защищает от деления на ноль, если текст состоял только из <PAD>
        pooled = masked_embedded.sum(dim=1) / mask.sum(dim=1).clamp(min=1) # (batch_size, embedding_dim)
        
        # Прогоняем через сеть: Linear -> ReLU -> Dropout -> Linear
        x = self.fc1(pooled)
        x = self.relu(x)
        x = self.dropout(x) # защищает от переобучения признаков
        
        x = self.fc2(x) # Выходной слой без Dropout
        
        return x
        
        
    
    
    
    
    