# em_train_word2vec.py

"""
Скрипт для обучения Word2Vec модели на спортивных текстах.
Сохраняет модель и словарь для дальнейшего использования.
"""
import sys
import os
import logging
import pandas as pd
import nltk
from gensim.models import Word2Vec

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                
from b_scr import ba_config

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Обучение Word2Vec модели")
    
    # 1. Загружаем очищенные данные
    logger.info("Загрузка данных")
    df = pd.read_csv(ba_config.PROCESSED_DATA_PATH)
    logger.info(f" - Загружено {len(df)} текстов")
    
    # 2. Токенизация текстов
    logger.info("Токенизация текстов")
    df["tokens"] = df["clean_text"].str.split()
    
    # Удаляем пустые токены
    df["tokens"] = df["tokens"].apply(lambda tokens: [t for t in tokens if t.strip()])
    
    logger.info(f" Токенизация завершена. Среднее количество токенов: {df['tokens'].apply(len).mean():.1f}")
    
    # 3. Инициализация и обучение Word2Vec
    logger.info("\n Инициализация Word2Vec модели")
    logger.info("Используемые гиперпараметры:")
    logger.info(f"  vector_size: {ba_config.W2V_VECTOR_SIZE}")
    logger.info(f"  window: {ba_config.W2V_WINDOW}")
    logger.info(f"  min_count: {ba_config.W2V_MIN_COUNT}")
    sg_type = "Skip-Gram" if ba_config.W2V_SG == 1 else "CBOW"
    logger.info(f"  sg: {ba_config.W2V_SG} ({sg_type})")
    total_epochs = ba_config.W2V_EPOCHS_INITIAL + ba_config.W2V_EPOCHS_ADDITIONAL
    logger.info(f"  total_epochs: {total_epochs} ({ba_config.W2V_EPOCHS_INITIAL} + {ba_config.W2V_EPOCHS_ADDITIONAL})")
    
    # Инициализация модели
    model_w2v = Word2Vec(
        sentences=df["tokens"],
        vector_size=ba_config.W2V_VECTOR_SIZE,
        window=ba_config.W2V_WINDOW,
        min_count=ba_config.W2V_MIN_COUNT,
        sg=ba_config.W2V_SG,
        workers=ba_config.W2V_WORKERS,
        epochs=ba_config.W2V_EPOCHS_INITIAL,
        seed=ba_config.W2V_SEED
    )
    
    # Дообучение
    logger.info("- Дообучение модели")
    model_w2v.train(
        df["tokens"],
        total_examples=model_w2v.corpus_count,
        epochs=ba_config.W2V_EPOCHS_ADDITIONAL
    )
    
    # 4. Сохранение модели
    logger.info("\n- Сохранение модели")
    os.makedirs(ba_config.MODELS_PATH, exist_ok=True)
    model_save_path = os.path.join(ba_config.MODELS_PATH, "word2vec_sports.model")
    model_w2v.save(model_save_path)
    logger.info(f" - Модель сохранена в: {model_save_path}")
    
    # 5. Статистика модели
    vocab_size = len(model_w2v.wv.index_to_key)
    logger.info(f"\n- Статистика модели:")
    logger.info(f"  Размер словаря: {vocab_size} слов")
    logger.info(f"  Размерность векторов: {model_w2v.vector_size}")
    
    # 6. Примеры похожих слов
    logger.info("\n- Примеры семантической близости (топ-3):")
    test_words = ['футбол', 'хоккей', 'теннис', 'игрок', 'матч']
    for word in test_words:
        if word in model_w2v.wv:
            similar = model_w2v.wv.most_similar(word, topn=3)
            similar_str = ', '.join([f"{w} ({s:.2f})" for w, s in similar])
            logger.info(f"  {word:10s} - {similar_str}")
        else:
            logger.info(f"  {word:10s} - (нет в словаре, min_count={ba_config.W2V_MIN_COUNT}?)")
            
    logger.info(f"Обучение завершено! Модель готова.")
    
if __name__ == "__main__":
    main()
    
            
    
    
    
    




