# bf_inference.py

"""
Модуль для инференса (предсказания) на новых данных.
Загружает сохраненные артефакты (пайплайны и модели) и делает предсказания.
"""

import logging
import os
import joblib
from catboost import CatBoostClassifier

from . import ba_config
from .bc_preprocessing import preprocess_text

logger = logging.getLogger(__name__)

def predict_single_text(raw_text: str, model_type: str = 'logistic_regression'):
    """
    Принимает сырой текст и возвращает предсказанную категорию и уверенность модели.
    
    Args:
        raw_text: Исходный текст от пользователя.
        model_type: 'logistic_regression' или 'catboost_svd'.
        
    Returns:
        tuple: (predicted_category, confidence_score)
    """
    if not raw_text or not isinstance(raw_text, str):
        return "Ошибка: пустой или некорректный текст", 0.0
    
    # 1. Препроцессинг
    clean_text = preprocess_text(raw_text)
    if not clean_text.strip():
        return "Неизвестно (текст полностью состоит из стоп-слов)", 0.0
        
    # 2. Определение путей к артефактам
    if model_type == 'logistic_regression':
        pipeline_path = os.path.join(ba_config.MODELS_PATH, "tfidf_vectorizer.joblib")
        model_path = os.path.join(ba_config.MODELS_PATH, "logistic_regression.joblib")
    elif model_type == 'catboost_svd':
        pipeline_path = os.path.join(ba_config.MODELS_PATH, "tfidf_svd_pipeline.joblib")
        model_path = os.path.join(ba_config.MODELS_PATH, "catboost_model_svd.cbm")
    else:
        raise ValueError(f"Неизвестный тип модели: {model_type}")
        
    # Проверка наличия файлов
    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        raise FileNotFoundError(f"Артефакты для {model_type} не найдены. Сначала обучите модель!")
    
    logger.info(f"Загрузка артефактов для {model_type}")
    
    # 3. Загрузка пайплайна и модели
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
        
    # 4. Векторизация текста
    X = pipeline.transform([clean_text])
    
    # 5. Предсказание и расчет уверенности
    probabilities = model.predict_proba(X)[0]
    
    # Находим индекс класса с максимальной вероятностью
    pred_idx = probabilities.argmax()
    
    # Получаем название класса и его вероятность
    predicted_category = model.classes_[pred_idx]
    confidence = probabilities[pred_idx]
    
    return predicted_category, confidence
    
    
    