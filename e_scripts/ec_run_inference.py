# ec_run_inference.py

"""
Скрипт для тестирования инференса обученных моделей.
"""
import sys
import os
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr.bf_inference import predict_single_text

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def main():
    # Тестовые тексты (имитация запросов от пользователя)
    test_texts = [
        "Футбольный матч завершился победой сборной России со счетом 3:1",
        "Теннисист выиграл турнир Большого шлема в упорной борьбе",
        "Новый болид Формулы-1 показал рекордную скорость на трассе",
        "Лыжник занял первое место на зимних соревнованиях в Сочи",
        "Вчера играл в настольную игру с друзьями, было весело",
        "Вчера сделали с друзьями заказ в этом кафе, было весело, еда вкусная, обслуживание хорошее",
        "33 33 33" # Тест на мусорный ввод
    ]
    
    models_to_test = ['logistic_regression', 'catboost_svd']
    
    for model_type in models_to_test:
        logger.info(f"Тестирование: {model_type.upper().replace('_', ' ')}")
        
        for text in test_texts:
            try:
                category, confidence =predict_single_text(text, model_type=model_type)
                logger.info(f" Текст: \"{text[:60]}{'...' if len(text)>60 else ''}\"")
                logger.info(f" Категория: {category} | Уверенность: {confidence:.2%}\n")
            except Exception as e:
                logger.error(f"Ошибка при обработке текста: {e}")

if __name__ == "__main__":
    main()



