# bh_inference_embeddings.py

"""
Модуль для инференса на новых данных с использованием предобученных эмбеддингов (E5).
"""
import os
import logging
import numpy as np
import torch
import joblib
from sentence_transformers import SentenceTransformer
from catboost import CatBoostClassifier

from . import ba_config
from .bc_preprocessing import preprocess_text

logger = logging.getLogger(__name__)

# Глобальная инициализация модели эмбеддингов
# Загружаем ОДИН раз при импорте модуля, чтобы не тратить время на каждый запрос
logger.info("Инициализация модели эмбеддингов (multilingual-e5-small)...")

# Определяем устройство (CPU или GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f" Модель эмбеддингов будет работать на устройстве: {device.upper()}")

embedding_model = SentenceTransformer("intfloat/multilingual-e5-small", device=device)
logger.info(" Модель эмбеддингов готова к работе.")

def predict_single_text_embeddings(raw_text: str, model_type: str = 'logistic_regression'):
    """
    Принимает сырой текст, векторизует его через E5 и делает предсказание.
    
    Args:
        raw_text: Исходный текст.
        model_type: 'logistic_regression' или 'catboost'.
        
    Returns:
        tuple: (predicted_category, confidence_score)
    """
    if not raw_text or not isinstance(raw_text, str):
            return "Ошибка: пустой или некорректный текст", 0.0
        
    # 1. Препроцессинг
    clean_text = preprocess_text(raw_text)
    if not clean_text.strip():
        return "Неизвестно (текст полностью состоит из стоп-слов)", 0.0
    
    # 2. Векторизация через E5
    text_with_prefix = f"passage: {clean_text}"
    embedding = embedding_model.encode(
        text_with_prefix,
        normalize_embeddings=True
    )
    
    # Модели sklearn/catboost ожидают 2D массив (batch_size, n_features)
    X = embedding.reshape(1, -1)
    
    # 3. Определение путей и загрузка модели
    if model_type == 'logistic_regression':
        model_path = os.path.join(ba_config.MODELS_PATH, "logistic_regression.joblib")
        model = joblib.load(model_path)
    elif model_type == 'catboost':
        model_path = os.path.join(ba_config.MODELS_PATH, "catboost_model.cbm")
        model = CatBoostClassifier()
        model.load_model(model_path)
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
    
    # 4. Предсказание и расчет уверенности
    probabilities = model.predict_proba(X)[0]
    pred_idx = np.argmax(probabilities)
    
    predicted_category = model.classes_[pred_idx]
    confidence = probabilities[pred_idx]
    
    return predicted_category, confidence
    







