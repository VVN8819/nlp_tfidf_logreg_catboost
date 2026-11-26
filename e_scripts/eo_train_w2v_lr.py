# eo_train_w2v_lr.py

"""
Скрипт для обучения Logistic Regression на Word2Vec эмбеддингах.
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config
from b_scr.be_models import train_logistic_regression_w2v

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Обучение Logistic Regression на Word2Vec эмбеддингах")
    
    # 1. Загружаем данные с эмбеддингами
    logger.info("Загрузка данных")
    w2v_emb_path = os.path.join(os.path.dirname(ba_config.PROCESSED_DATA_PATH), "w2v_embedded_train.csv")

    if not os.path.exists(w2v_emb_path):
        logger.error(f"- Файл не найден: {w2v_emb_path}")
        logger.error("   Сначала запустите em_extract_w2v_embeddings.py")
        return
    
    df = pd.read_csv(w2v_emb_path)
    logger.info(f"- Загружено {len(df)} текстов с эмбеддингами")
    
    # 2. Разделяем на признаки и метки
    logger.info("\nПодготовка данных")
    emb_columns = [col for col in df.columns if col.startswith('w2v_')]
    X = df[emb_columns].values
    y = df['category'].values
    
    logger.info(f"   Признаки: {X.shape}")
    logger.info(f"   Метки: {len(y)}")
    
    # 3. Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=ba_config.TEST_SIZE,
        random_state=ba_config.RANDOM_SEED,
        stratify=y
    )
    
    logger.info(f"- Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 4. Обучаем Logistic Regression
    logger.info("\n- Обучение Logistic Regression")
    lr_model, lr_acc = train_logistic_regression_w2v(X_train, y_train, X_val, y_val)
    
    # 5. Финальный вывод
    logger.info(" Итоговые результаты Модель: Logistic Regression на Word2Vec эмбеддингах")
    logger.info(f"Размерность признаков: {X.shape[1]}")
    logger.info(f"Logistic Regression Accuracy: {lr_acc:.4f} ({lr_acc*100:.2f}%)")
    
if __name__ == "__main__":
    main()




