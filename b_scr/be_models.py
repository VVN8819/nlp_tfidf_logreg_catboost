# be_models.py

"""
Модуль для обучения моделей классификации.
Содержит функции для обучения Logistic Regression и CatBoost.
"""

import logging
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

from . import ba_config

logger = logging.getLogger(__name__)

def evaluate_and_save_model(model, model_name, X_val, y_val, model_path):
    """
    Универсальная функция для оценки и сохранения модели.
    """
    logger.info(f"Оценка модели {model_name}")
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    logger.info(f" - {model_name} - Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    
    # Classification report
    report = classification_report(y_val, y_pred, zero_division=0)
    logger.info(f"Classification Report:\n{report}")
    
    # Сохранение
    os.makedirs(ba_config.MODELS_PATH, exist_ok=True)
    if model_name == 'CatBoost':
        model.save_model(model_path)
    else:
        joblib.dump(model, model_path)
    logger.info(f" Модель сохранена в: {model_path}")
    
    return acc

def train_logistic_regression(X_train, y_train, X_val, y_val):
    """
    Обучает Logistic Regression на TF-IDF признаках.
    """
    logger.info(" Обучение Logistic Regression")
    
    model = LogisticRegression(
        random_state=ba_config.RANDOM_SEED,
        max_iter=1000,
        n_jobs=1,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    
    model_path = os.path.join(ba_config.MODELS_PATH, "logistic_regression.joblib")
    acc = evaluate_and_save_model(model, "Logistic Regression", X_val, y_val, model_path)
    
    return model, acc

def train_catboost(X_train, y_train, X_val, y_val):
    """
    Обучает CatBoost на TF-IDF признаках.
    """
    logger.info(" Обучение CatBoost")
    
    model = CatBoostClassifier(
        iterations=500, # Увеличиваем количество деревьев
        learning_rate=0.05, # Уменьшаем шаг обучения для стабильности
        depth=4, # мельче деревья
        l2_leaf_reg=3, # Регуляризация против переобучения
        auto_class_weights='Balanced',   # балансировка 13 классов
        loss_function='MultiClass',
        random_seed=ba_config.RANDOM_SEED,
        verbose=50
    )
    logger.info("Начало обучения:")
    model.fit(X_train, y_train)
    
    model_path = os.path.join(ba_config.MODELS_PATH, "catboost_model.cbm")
    acc = evaluate_and_save_model(model, "CatBoost", X_val, y_val, model_path)
    
    return model, acc
    
    
    
    