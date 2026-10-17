# ea_train_logistic_regression.py

"""
Скрипт для обучения Logistic Regression на TF-IDF признаках.
"""
import sys
import os
import pandas as pd
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config
from b_scr.bd_feature_extraction import create_and_save_tfidf
from b_scr.be_models import train_logistic_regression

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Обучение Logistic Regression")
    
    # 1. Загружаем очищенные данные
    logger.info(" - Загрузка очищенных данных")
    df = pd.read_csv(ba_config.PROCESSED_DATA_PATH)
    
    # 2. Разделяем на train/val
    from sklearn.model_selection import train_test_split
    train_df, val_df = train_test_split(
        df,
        test_size=ba_config.TEST_SIZE,
        random_state=ba_config.RANDOM_SEED,
        stratify=df[ba_config.TARGET_COL]
    )
    
    # 3. Создаем TF-IDF признаки
    logger.info(" - Создание TF-IDF признаков")
    vectorizer, X_train, X_val = create_and_save_tfidf(
        train_df['clean_text'],
        val_df['clean_text']
    )
    
    # 4. Обучаем модель
    model, accuracy = train_logistic_regression(
        X_train, train_df[ba_config.TARGET_COL],
        X_val, val_df[ba_config.TARGET_COL]
    )
    
    logger.info(f"Обучение завершено! Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()
    
    