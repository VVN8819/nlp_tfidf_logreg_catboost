# ba_config.py

"""
Конфигурация проекта Sports Text Classifier.
Единый источник истины для всех настроек.
"""
import os

# получаем абсолютный путь к корню проекта.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ПУТИ К ДАННЫМ
DATA_DIR = os.path.join(BASE_DIR, "a_data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "aa_raw", "train.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "ab_processed", "preprocessed_train.csv")

# НАЗВАНИЯ КОЛОНОК
ID_COL = "oid"
TEXT_COL = "text"
TARGET_COL = "category"

# КАТЕГОРИИ
ALL_CATEGORIES = [
    "athletics", "autosport", "basketball", "boardgames", "esport",
    "extreme", "football", "hockey", "martial arts", "motosport",
    "tennis", "volleyball", "winter_sport"
]
N_CLASSES = len(ALL_CATEGORIES)

# ПАРАМЕТРЫ ОБУЧЕНИЯ И РАЗДЕЛЕНИЯ
RANDOM_SEED = 42
TEST_SIZE = 0.2


