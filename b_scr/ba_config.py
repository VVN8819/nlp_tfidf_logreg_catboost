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

# ПУТИ К МОДЕЛЯМ И АРТЕФАКТАМ
MODELS_PATH = os.path.join(BASE_DIR, "d_models")

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

# Deep learning ПАРАМЕТРЫ
MAX_SEQ_LENGTH = 128  # Из EDA (95th percentile)
MAX_VOCAB_SIZE = 10000  # Ограничение словаря
EMBEDDING_DIM = 100  # Размерность word embeddings

# Special tokens
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'
PAD_IDX = 0
UNK_IDX = 1

# Обучение DL моделей
DL_BATCH_SIZE = 64
DL_EPOCHS = 20
DL_LEARNING_RATE = 0.001
DL_HIDDEN_DIM = 256
DL_DROPOUT = 0.5

PLOTS_PATH = os.path.join(BASE_DIR, "plots")
