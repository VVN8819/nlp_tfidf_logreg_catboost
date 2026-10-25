# ed_extract_embeddings.

"""
Скрипт для извлечения эмбеддингов из очищенных текстов.
"""
import sys
import os
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))#

from b_scr import ba_config
from b_scr.bg_embedding_extractor import create_and_save_embeddings

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Извлечение эмбеддингов")
    
    input_path = ba_config.PROCESSED_DATA_PATH
    output_path = os.path.join(os.path.dirname(ba_config.PROCESSED_DATA_PATH), "embedded_train.csv")
    
    logger.info(f"Вход: {input_path}")
    logger.info(f"Выход: {output_path}")
    
    df = create_and_save_embeddings(
        input_csv_path=input_path,
        output_csv_path=output_path,
        model_name="intfloat/multilingual-e5-small",
        batch_size=32
    )
    
    logger.info("Извлечение эмбеддингов завершено")
    
if __name__ == "__main__":
    main()
    


