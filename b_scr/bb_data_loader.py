# bb_data_loader.py

"""
Модуль для загрузки и базовой валидации сырых данных.
"""
import pandas as pd
import logging
import os
from datetime import datetime
from sklearn.model_selection import train_test_split

from . import ba_config

# Создаем папку для логов (если нет)
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

# Имя файла с датой и временем
log_filename = os.path.join(LOGS_DIR, f"data_loader_{datetime.now().strftime('%Y%m%d')}.log")

# Логирование вместо обычных print
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # Сохраняем в файл
        logging.StreamHandler()  # Выводим в консоль
    ]
)

logging.info(f"Логи сохраняются в: {log_filename}")

def load_and_validate_data(filepath: str) -> pd.DataFrame:
    """
    Загружает CSV-файл и проводит базовую проверку целостности данных.
    
    Args:
        filepath: Путь к CSV-файлу
        
    Returns:
        pd.DataFrame: Очищенный от критических ошибок DataFrame
    """
    
    logging.info(f"Загрузка данных из: {filepath}")
    
    # 1. Загрузка
    df = pd.read_csv(filepath, encoding='utf-8')
    logging.info(f"Исходный размер данных: {df.shape}")
    
    # 2. Проверка наличия нужных колонок
    required_cols = [ba_config.ID_COL, ba_config.TEXT_COL, ba_config.TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Критическая ошибка! В файле отсутствуют колонки: {missing_cols}")
    
    # 3. Удаление полных дубликатов
    initial_len = len(df)
    df = df.drop_duplicates(subset=[ba_config.TEXT_COL])
    if len(df) < initial_len:
        logging.info(f"Удалено {initial_len - len(df)} полных дубликатов текста.")
    
    # 4. Удаление строк с пропусками в ключевых колонках
    df = df.dropna(subset=[ba_config.TEXT_COL, ba_config.TARGET_COL])
    
    # 5. Нормализация названий категорий (пробелы лишние)
    df[ba_config.TARGET_COL] = df[ba_config.TARGET_COL].str.lower().str.strip()
    
    logging.info(f"Данные успешно загружены и валидированы. Итоговый размер: {df.shape}")
    return df

def get_train_val_splits(df: pd.DataFrame):
    """
    Разделяет DataFrame на обучающую и валидационную выборки.
    Использует stratify для сохранения пропорций классов (защита от дисбаланса).
    """
    logging.info("Разделение данных на train и val")
    
    train_df, val_df = train_test_split(
        df,
        test_size=ba_config.TEST_SIZE,
        random_state=ba_config.RANDOM_SEED,
        stratify=df[ba_config.TARGET_COL]
    )
    
    logging.info(f"Размер Train: {train_df.shape}, Размер Val: {val_df.shape}")
    return train_df, val_df

