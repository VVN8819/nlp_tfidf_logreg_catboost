# ee_train_embeddings.py#

"""
Скрипт для обучения моделей на плотных эмбеддингах (E5).
"""

import sys
import os
import pandas as pd
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#

from b_scr import ba_config
from b_scr.be_models import train_logistic_regression, train_catboost

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info(" Обучение моделей на плотных эмбеддингах (multilingual-e5-small)")
    
    # 1. Загружаем данные с эмбеддингами
    embeddings_path = os.path.join(os.path.dirname(ba_config.PROCESSED_DATA_PATH), "embedded_train.csv")
    logger.info(f" Загрузка данных из: {embeddings_path}")
    
    if not os.path.exists(embeddings_path):
        logger.error(f" Файл embedded_train.csv не найден! Сначала запустите скрипт извлечения эмбеддингов.")
        return
    
    df = pd.read_csv(embeddings_path)
    logger.info(f" Загружено строк: {len(df)}")
    
    # 2. Разделяем на train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=ba_config.TEST_SIZE,
        random_state=ba_config.RANDOM_SEED,
        stratify=df[ba_config.TARGET_COL]
    )
    
    # 3. Формируем X и y
    # X - это все колонки, начинающиеся с 'emb_' (их ровно 384)
    emb_cols = [col for col in df.columns if col.startswith('emb_')]
    logger.info(f" Количество признаков (эмбеддингов): {len(emb_cols)}")
    
    X_train = train_df[emb_cols].values
    y_train = train_df[ba_config.TARGET_COL].values
    
    X_val = val_df[emb_cols].values
    y_val = val_df[ba_config.TARGET_COL].values
    
    # 4. Обучаем Logistic Regression
    logger.info("1️. Обучение Logistic Regression на эмбеддингах")
    lr_model, lr_acc = train_logistic_regression(X_train, y_train, X_val, y_val)
    
    # 5. Обучаем CatBoost
    logger.info("2️. Обучение CatBoost на эмбеддингах")
    cb_model, cb_acc = train_catboost(X_train, y_train, X_val, y_val)
    
    # 6. Финальный вывод
    logger.info(" Итоговые результаты на эмбеддингах:")
    logger.info(f"Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    logger.info(f"CatBoost Accuracy:            {cb_acc:.4f} ({cb_acc*100:.2f}%)")
    
    if cb_acc > lr_acc:
        logger.info(" Гипотеза подтверждена: CatBoost превзошел Logistic Regression на плотных признаках!")
    else:
        logger.info(" LR показал себя не хуже или лучше на этих данных.")

if __name__ == "__main__":
    main()


