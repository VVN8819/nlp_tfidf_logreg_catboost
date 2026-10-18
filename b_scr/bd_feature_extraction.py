# bd_feature_extraction.py

"""
Модуль для извлечения признаков из текста с помощью TF-IDF и уменьшения размерности (SVD).
"""
import logging
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from . import ba_config

logger = logging.getLogger(__name__)

def create_and_save_tfidf_svd(train_text, val_text):
    """
    Создает пайплайн: TF-IDF векторизатор -> TruncatedSVD.
    Обучает его на train и трансформирует train и val.
    Сохраняет векторизатор на диск для будущего инференса.
    
    Args:
        train_texts: pd.Series с очищенным текстом для обучения
        val_texts: pd.Series с очищенным текстом для валидации
        
    Returns:
        vectorizer: Обученный TfidfVectorizer
        X_train: Разреженная матрица признаков для train
        X_val: Разреженная матрица признаков для val
    """
    # logger.info("Инициализация TfidfVectorizer")
    logger.info("Инициализация пайплайна TF-IDF + TruncatedSVD")
    
    # Настройки векторизатора, основанные на EDA
    vectorizer = TfidfVectorizer(
        max_features=10000, # Ограничиваем словарь топ-10000 слов
        ngram_range=(1, 2), # Учитываем отдельные слова и биграммы (пары слов)
        min_df=5, # Игнорируем слова, встречающиеся менее чем в 5 документах
        max_df=0.8, # Игнорируем слова, встречающиеся в более чем 80% документов
        sublinear_tf=True # Применяем логарифмическое масштабирование частот (улучшает качество)
    )
    
    # 2. Уменьшение размерности (SVD)
    svd = TruncatedSVD(
        n_components=500, # сжимаем 10 000 слов до 500 "скрытых тем"
        random_state=ba_config.RANDOM_SEED
    )
    
    # 3. Объединяем в пайплайн
    pipeline = Pipeline([
        ('tfidf', vectorizer),
        ('svd', svd)
    ])
    
    logger.info("Обучение векторизатора на тренировочных данных")
    # ВАЖНО: fit_transform только на train, чтобы не было утечки данных из val!
    #X_train = vectorizer.fit_transform(train_text)
    X_train = pipeline.fit_transform(train_text)
    
    logger.info("Трансформация валидационных данных")
    #X_val = vectorizer.transform(val_text)
    X_val = pipeline.transform(val_text)
    
    logger.info(f"Размер словаря (Vocabulary size): {len(vectorizer.vocabulary_)}")
    logger.info(f"Размерность после SVD: {X_train.shape[1]}")
    logger.info(f"Форма X_train: {X_train.shape}, Форма X_val: {X_val.shape}")
    
    # Сохраняем ВЕСЬ пайплайн в один файл
    os.makedirs(ba_config.MODELS_PATH, exist_ok=True)
    # Сохраняем векторизатор на диск (он понадобится для инференса)
    # vectorizer_path = os.path.join(ba_config.MODELS_PATH, "tfidf_vectorizer.joblib")
    pipeline_path = os.path.join(ba_config.MODELS_PATH, "tfidf_svd_pipeline.joblib")
    
    #joblib.dump(vectorizer, vectorizer_path)
    #logger.info(f"Векторизатор успешно сохранен в: {vectorizer_path}")
    
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Пайплайн успешно сохранен в: {pipeline_path}")
    
    return pipeline, X_train, X_val
    #return vectorizer, X_train, X_val


