# bi_data_loader_dl.py

"""
DataLoader для Deep Learning моделей.
Превращает текст в тензоры с padding и <UNK> токенами.
"""
import logging
import os
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pandas as pd

from . import ba_config
from .bc_preprocessing import preprocess_text

logger = logging.getLogger(__name__)

class Vocabulary:
    """
    Словарь для преобразования слов в индексы и обратно.
    Включает специальные токены <PAD> и <UNK>.
    """
    def __init__(self, max_size=None):
        self.word2idx = {
            ba_config.PAD_TOKEN: ba_config.PAD_IDX,
            ba_config.UNK_TOKEN: ba_config.UNK_IDX
        }
        self.idx2word = {
            ba_config.PAD_IDX: ba_config.PAD_TOKEN,
            ba_config.UNK_IDX: ba_config.UNK_TOKEN
        }
        self.max_size = max_size
        self.word_freq = Counter()
        
    def build_vocab(self, texts):
        """
        Строит словарь из списка текстов.
        Ограничивает размер max_size, оставляя только самые частые слова.
        """
        logger.info(f"Построение словаря из {len(texts)} текстов")
        
        # Считаем частоты слов
        for text in texts:
            words = text.split()
            self.word_freq.update(words)
            
        # Ограничиваем словарь
        if self.max_size:
            # Берем топ-(max_size - 2) слов, потому что 2 места заняты <PAD> и <UNK>
            most_common = self.word_freq.most_common(self.max_size - 2)
        else:
            most_common = self.word_freq.most_common()
            
        # Добавляем слова в словарь
        for word, freq in most_common:
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        logger.info(f" - Словарь построен: {len(self.word2idx)} слов")
        logger.info(f" - Покрытие: {sum(self.word_freq[w] for w in self.word2idx) / sum(self.word_freq.values()) * 100:.1f}%")
        
    def text_to_indices(self, text, max_length=None):
        """
        Превращает текст в список индексов с padding/truncation.
        """
        if max_length is None:
            max_length = ba_config.MAX_SEQ_LENGTH
            
        words = text.split()
        indices = []
        
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(ba_config.UNK_IDX)
                
        # Truncation (обрезка длинных текстов)
        if len(indices) > max_length:
            indices = indices[:max_length]
        
        # Padding (дополнение коротких текстов)
        while len(indices) < max_length:
            indices.append(ba_config.PAD_IDX)
            
        return indices
    
    def __len__(self):
        return len(self.word2idx)

class TextDataset(Dataset):
    """
    PyTorch Dataset для текстовых данных.
    """
    def __init__(self, texts, labels, vocabulary, max_length=None):
        """
        Args:
            texts: Список очищенных текстов
            labels: Список меток (строк или чисел)
            vocabulary: Объект Vocabulary
            max_length: Максимальная длина последовательности
        """
        self.texts = texts
        self.labels = labels
        self.vocabulary = vocabulary
        self.max_length = max_length or ba_config.MAX_SEQ_LENGTH
        
        # Кодируем метки в числа
        if isinstance(labels[0], str):
            self.label_classes = sorted(set(labels))
            self.label2idx = {label: idx for idx, label in enumerate(self.label_classes)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
            self.encoded_labels = [self.label2idx[label] for label in labels]
        else:
            self.encoded_labels = labels
            self.label_classes = sorted(set(labels))
            self.label2idx = {label: idx for idx, label in enumerate(self.label_classes)}
            self.idx2label = {idx: label for label, idx in self.label2idx.items()}
            
        logger.info(f" - Dataset: {len(texts)} текстов, {len(self.label_classes)} классов")
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.encoded_labels[idx]
        
        # Превращаем текст в индексы с padding
        indices = self.vocabulary.text_to_indices(text, self.max_length)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)
  
def create_dataloaders(train_texts, train_labels, val_texts, val_labels,
                       vocabulary=None, batch_size=None):
    """
    Создает DataLoader'ы для обучения и валидации.
    
    Returns:
        vocabulary: Построенный словарь
        train_loader: DataLoader для обучения
        val_loader: DataLoader для валидации
    """
    if batch_size is None:
        batch_size = ba_config.DL_BATCH_SIZE
        
    if vocabulary is None:
        vocabulary = Vocabulary(max_size=ba_config.MAX_VOCAB_SIZE)
        vocabulary.build_vocab(train_texts)
        
    # Создаем Dataset
    train_dataset = TextDataset(train_texts, train_labels, vocabulary)
    val_dataset = TextDataset(val_texts, val_labels, vocabulary)
    
    # Создаем DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True # Отбрасываем последний неполный батч
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f" - DataLoader созданы:")
    logger.info(f"   Train: {len(train_loader)} батчей по {batch_size}")
    logger.info(f"   Val: {len(val_loader)} батчей по {batch_size}")
    
    return vocabulary, train_loader, val_loader

def load_data_for_dl(csv_path=None):
    """
    Загружает данные из CSV и разделяет на train/val.
    """
    if csv_path is None:
        csv_path = ba_config.PROCESSED_DATA_PATH
        
    logger.info(f"Загрузка данных из: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Разделяем на train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=ba_config.TEST_SIZE,
        random_state=ba_config.RANDOM_SEED,
        stratify=df[ba_config.TARGET_COL]
    )
    
    logger.info(f" - Данные загружены: {len(train_df)} train, {len(val_df)} val")
    
    return (
        train_df['clean_text'].tolist(),
        train_df[ba_config.TARGET_COL].tolist(),
        val_df['clean_text'].tolist(),
        val_df[ba_config.TARGET_COL].tolist(),
    )




