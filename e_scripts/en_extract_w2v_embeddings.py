# en_extract_w2v_embeddings.py

"""
Скрипт для извлечения Word2Vec эмбеддингов из текстов.
Усредняет векторы известных слов для получения вектора текста.
"""
import sys
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Извлечение Word2Vec эмбеддингов")
    
    # 1. Загружаем обученную модель Word2Vec
    logger.info("Загрузка модели Word2Vec")
    model_path = os.path.join(ba_config.MODELS_PATH, "word2vec_sports.model")
    model_w2v = Word2Vec.load(model_path)
    
    logger.info(f"- Модель загружена. Размер словаря: {len(model_w2v.wv.index_to_key)}")
    logger.info(f"  Размерность векторов: {model_w2v.vector_size}")
    
    # 2. Загружаем очищенные данные
    logger.info("\nЗагрузка данных")
    df = pd.read_csv(ba_config.PROCESSED_DATA_PATH)
    logger.info(f"- Загружено {len(df)} текстов")
    
    # 3. Извлекаем эмбеддинги
    logger.info("\nИзвлечение эмбеддингов (усреднение векторов слов)")
    
    # Преобразуем колонку в список для быстрой итерации
    texts = df['clean_text'].to_list()
    embeddings = []
    
    for text in tqdm(texts, desc="Обработка текстов", unit="текст"):
        words = text.split()
            
        # Получаем векторы ТОЛЬКО для слов, которые есть в словаре
        vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]
        
        if len(vectors) == 0:
            # Если текст пустой - возвращаем нулевой вектор
            embeddings.append(np.zeros(model_w2v.vector_size))
        else:
            # Усредняем только известные векторы
            embeddings.append(np.mean(vectors, axis=0))
        
    # Преобразуем список векторов в матрицу
    embeddings_matrix = np.array(embeddings)
    logger.info(f"- Эмбеддинги извлечены: {embeddings_matrix.shape}")
    
    # 4. Создаем DataFrame с эмбеддингами
    logger.info("\nСоздание DataFrame")
    emb_columns = [f'w2v_{i}' for i in range(model_w2v.vector_size)]
    emb_df = pd.DataFrame(embeddings_matrix, columns=emb_columns)
    
    # Добавляем oid и category
    emb_df.insert(0, 'category', df['category'].values)
    emb_df.insert(0, 'oid', df['oid'].values)
    
    logger.info(f"- DataFrame создан: {emb_df.shape}")
    
    # 5. Сохраняем в CSV
    logger.info("\n- Сохранение эмбеддингов")
    os.makedirs(os.path.dirname(ba_config.PROCESSED_DATA_PATH), exist_ok=True)
    w2v_emb_path = os.path.join(os.path.dirname(ba_config.PROCESSED_DATA_PATH), "w2v_embedded_train.csv")
    emb_df.to_csv(w2v_emb_path, index=False)
    logger.info(f"- Эмбеддинги сохранены в: {w2v_emb_path}")
    
    # 6. Статистика
    file_size_mb = os.path.getsize(w2v_emb_path) / (1024 * 1024)
    logger.info(f"\n- Статистика:")
    logger.info(f"  Количество текстов: {len(emb_df)}")
    logger.info(f"  Размерность эмбеддингов: {model_w2v.vector_size}")
    logger.info(f"  Размер файла: {file_size_mb:.2f} МБ")
    logger.info(f"  Колонки: oid, category, w2v_0...w2v_{model_w2v.vector_size - 1}")
    
    logger.info("\n Извлечение Word2Vec эмбеддингов завершено")
    
if __name__ == "__main__":
    main()
    



