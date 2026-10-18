# eb_train_catboost.py

"""
Скрипт для обучения CatBoost на TF-IDF признаках.
"""
import sys
import os
import pandas as pd
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config
#from b_scr.bd_feature_extraction import create_and_save_tfidf
from b_scr.bd_feature_extraction import create_and_save_tfidf_svd
from b_scr.be_models import train_catboost

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Обучение CatBoost")
    
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
    # 3. Создаем признаки через новый пайплайн
    #logger.info(" - Создание TF-IDF признаков")
    logger.info(" - Создание TF-IDF + SVD признаков")
    #vectorizer, X_train, X_val = create_and_save_tfidf(
    pipeline, X_train, X_val = create_and_save_tfidf_svd(
        train_df['clean_text'],
        val_df['clean_text']
    )
    
    # 4. Обучаем модель
    logger.info(" - Обучение CatBoost")
    model, accuracy = train_catboost(
        X_train, train_df[ba_config.TARGET_COL],
        X_val, val_df[ba_config.TARGET_COL]
    )
    
    logger.info(f"Обучение завершено! Accuracy: {accuracy:.4f}")
    
if __name__ == "__main__":
    main()
    
    
    