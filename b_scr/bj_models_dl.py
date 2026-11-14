# bj_models_dl.py

"""
Архитектуры Deep Learning моделей для классификации текстов (FCN, RNN, CNN).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from . import ba_config

class TextFCN(nn.Module):
    """
    Fully Connected Network (Многослойный перцептрон) для текстов.
    Архитектура: Embedding -> Mean Pooling (с маской) -> FC Layers -> Output
    """
    def __init__(self, vocab_size, embedding_dim, num_classes, hidden_dim=256, dropout=0.5):
        super(TextFCN, self).__init__()
        
        # 1. Embedding слой
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=ba_config.PAD_IDX
        )
        
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
    
class Attention(nn.Module):
    """
    Механизм Attention для выделения важных слов в последовательности.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_out):
        # lstm_out: (batch_size, seq_length, hidden_dim)
        attn_weights = self.attention_weights(lstm_out) # (batch_size, seq_length, 1)
        attn_weights = torch.softmax(attn_weights, dim=1) # Нормализуем веса по длине последовательности
        
        # Умножаем веса на исходные векторы и суммируем
        context_vec = torch.sum(attn_weights * lstm_out, dim=1) # (batch_size, hidden_dim)
        return context_vec
    
class TextBiLSTM(nn.Module):
    """
    Двунаправленная RNN (BiLSTM) с механизмом Attention и Dropout.
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes, dropout_rate=0.3):
        super().__init__()
        
        # Щит 1: Embedding с ОБЯЗАТЕЛЬНЫМ padding_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=ba_config.PAD_IDX
        )
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Щит 2: BiLSTM. dropout применяется между слоями, если num_layers > 1
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if num_layers > 1 else 0.0
        )
        
        # Attention принимает размерность * 2, так как LSTM двунаправленный
        self.attention = Attention(hidden_dim *2)
        
        # Щит 3: Dropout перед финальным классификатором
        self.fc_dropout = nn.Dropout(dropout_rate)
        
        # Финальный слой (Судья)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_length)
        
        # 1. Превращаем индексы слов в векторы
        x = self.embedding(x)
        
        # 2. Применяем Dropout к эмбеддингам
        x = self.embed_dropout(x)
        
        # 3. Пропускаем через BiLSTM
        # output: (batch_size, seq_length, hidden_dim * 2)
        output, (hidden, cell) = self.bilstm(x)
        
        # 4. Attention выделяет контекстный вектор, взвешивая важные слова
        context_vec = self.attention(output) # (batch_size, hidden_dim * 2)
        
        # 5. Применяем Dropout перед классификацией
        context_vec = self.fc_dropout(context_vec)
        
        # 6. Финальное предсказание
        out = self.fc(context_vec) # (batch_size, num_classes)
        
        return out
        
class TextCNN(nn.Module):
    """
    Одномерная свёрточная сеть (CNN-1D) для классификации текстов.
    Архитектура: Embedding -> Multi-scale Conv1d -> Global Max Pooling -> FC -> Output
    """
    def __init__(self, vocab_size, embedding_dim, num_classes,
                 filter_sizes, num_filters, dropout_rate=0.5):
        super().__init__()
        
        # Щит 1: Embedding с padding_idx
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx=ba_config.PAD_IDX
        )
        self.embed_dropout = nn.Dropout(dropout_rate)
        
        # Множество сверточных слоев с разными размерами ядер
        # Каждый фильтр ищет паттерны разной длины (3, 4, 5 слов)
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=embedding_dim,
                    out_channels=num_filters,
                    kernel_size=fs
                ),
                nn.ReLU(),
                nn.Dropout(dropout_rate) # после каждой свертки
            ) for fs in filter_sizes
        ])
        
        # Global Max Pooling + финальный классификатор
        # Входная размерность: num_filters * len(filter_sizes)
        self.fc_dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
        
    def forward(self, x):
        # x: (batch_size, seq_length)

        # 1. Превращаем индексы слов в векторы
        embedded = self.embedding(x)
        
        # 2. Применяем Dropout к эмбеддингам
        embedded = self.embed_dropout(embedded)
        
        # 3. Conv1d ожидает (batch, channels, seq_length)
        # Поэтому меняем оси местами
        embedded = embedded.permute(0, 2, 1) # (batch_size, embedding_dim, seq_length)
        
        # 4. Применяем свертки с разными размерами ядер и Global Max Pooling
        pooled_outputs = []
        for conv in self.convs:
            # conv_out: (batch_size, num_filters, seq_length - kernel_size + 1)
            conv_output = conv(embedded)
            # Global Max Pooling - берем максимум по всей длине последовательности
            # Это аналог "найти самое сильное срабатывание фильтра в любом месте текста"
            pooled = F.max_pool1d(conv_output, conv_output.shape[2]).squeeze(2)
            pooled_outputs.append(pooled)
            
        # 5. Конкатенируем результаты всех свёрток
        # cat: (batch_size, num_filters * len(filter_sizes))
        cat = torch.cat(pooled_outputs, dim=1)
        
        # 6. Применяем Dropout перед классификацией
        cat = self.fc_dropout(cat)
        
        # 7. Финальное предсказание
        out = self.fc(cat) # (batch_size, num_classes)
        
        return out
        
        
    
    
    