# bg_embedding_extractor.py

"""
Модуль для извлечения плотных семантических эмбеддингов из текста
с использованием предобученной модели multilingual-e5-small.

Модель: https://huggingface.co/intfloat/multilingual-e5-small
Размерность эмбеддинга: 384
Языки: 100+ (включая русский)
"""

import logging
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import ba_config

logger = logging.getLogger(__name__)

def load_embedding_model(model_name: str = "intfloat/multilingual-e5-small"):
    """
    Загружает предобученную модель для извлечения эмбеддингов.
    
    Args:
        model_name: Название модели с HuggingFace
        
    Returns:
        SentenceTransformer: Загруженная модель
    """
    logger.info(f"Загрузка модели {model_name}")
    
    # Определяем устройство (CPU или GPU)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Используется устройство: {device}")
    
    # Загружаем модель
    model = SentenceTransformer(model_name, device=device)
    
    logger.info(f" Модель загружена. Размерность эмбеддинга: {model.get_sentence_embedding_dimension()}")
    return model

def extract_embeddings_batch(
    model: SentenceTransformer,
    texts: list,
    batch_size: int = 32,
    show_progress: bool = True
) -> np.ndarray:
    """
    Извлекает эмбеддинги из списка текстов батчами для экономии памяти.
    
    Args:
        model: Загруженная модель SentenceTransformer
        texts: Список текстов для кодирования
        batch_size: Размер батча (оптимально 32-64 для CPU)
        show_progress: Показывать прогресс-бар
        
    Returns:
        np.ndarray: Матрица эмбеддингов формы (len(texts), 384)
    """
    logger.info(f"Начало извлечения эмбеддингов для {len(texts)} текстов")
    logger.info(f"Размер батча: {batch_size}")
    
    # Модель E5 требует префикс "passage: " для лучшей работы
    prefixed_texts = [f"passage: {text}" for text in texts]
    
    # Извлекаем эмбеддинги с прогресс-баром
    embeddings = []
    
    iterator = tqdm(
        range(0, len(prefixed_texts), batch_size),
        desc="Извлечение эмбеддингов",
        unit="batch",
        disable=not show_progress
    )
    
    for i in iterator:
        batch = prefixed_texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False, # Отключаем встроенный прогресс, используем свой
            normalize_embeddings=True # Нормализуем векторы для косинусного сходства
        )
        embeddings.append(batch_embeddings)
        
    # Объединяем все батчи в одну матрицу
    embeddings_matrix = np.vstack(embeddings)
    
    logger.info(f" Извлечено эмбеддингов: {embeddings_matrix.shape}")
    return embeddings_matrix

def create_and_save_embeddings(
    input_csv_path: str,
    output_csv_path: str,
    model_name: str = "intfloat/multilingual-e5-small",
    batch_size: int = 32
):
    """
    Полный пайплайн: загрузка данных - извлечение эмбеддингов - сохранение.
    
    Args:
        input_csv_path: Путь к файлу с очищенными текстами (preprocessed_train.csv)
        output_csv_path: Путь для сохранения файла с эмбеддингами
        model_name: Название модели
        batch_size: Размер батча
        
    Returns:
        pd.DataFrame: DataFrame с эмбеддингами
    """
    logger.info("Извлечение эмбеддингов")
    
    # 1. Загружаем очищенные данные
    logger.info(f"Загрузка данных из: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    logger.info(f"Загружено строк: {len(df)}")
    
    # Проверяем наличие нужных колонок
    if ba_config.TEXT_COL not in df.columns:
        raise ValueError(f"Колонка '{ba_config.TEXT_COL}' не найдена в файле!")

    # 2. Загружаем модель
    model = load_embedding_model(model_name)
    
    # 3. Извлекаем эмбеддинги
    embeddings_matrix = extract_embeddings_batch(
        model=model,
        texts=df[ba_config.TEXT_COL].tolist(),
        batch_size=batch_size,
        show_progress=True
    )
    
    # 4. Создаем DataFrame с эмбеддингами
    # Формат: oid, category, emb_0, emb_1, ..., emb_383
    embedding_cols = [f"emb_{i}" for i in range(embeddings_matrix.shape[1])]
    embeddings_df = pd.DataFrame(embeddings_matrix, columns=embedding_cols)
    
    # Добавляем метаданные (oid и category)
    result_df = pd.concat([
        df[[ba_config.ID_COL, ba_config.TARGET_COL]].reset_index(drop=True),
        embeddings_df.reset_index(drop=True)
    ], axis=1)
    
    # 5. Сохраняем результат
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    logger.info(f" Эмбеддинги сохранены в: {output_csv_path}")
    logger.info(f"Размер файла: {os.path.getsize(output_csv_path) / 1024 / 1024:.2f} МБ")
    
    # 6. Статистика
    logger.info("\n Статистика:")
    logger.info(f"  Количество текстов: {len(result_df)}")
    logger.info(f"  Размерность эмбеддинга: {embeddings_matrix.shape[1]}")
    logger.info(f"  Колонки: {ba_config.ID_COL}, {ba_config.TARGET_COL}, emb_0...emb_383")
    
    return result_df
    


