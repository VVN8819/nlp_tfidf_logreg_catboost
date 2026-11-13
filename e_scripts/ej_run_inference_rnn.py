# ej_run_inference_rnn.py

"""
Скрипт для инференса Recurrent neural network (RNN).
Загружает обученную модель и словарь, делает предсказания на новых текстах.
"""
import sys
import os
import torch
import torch.nn.functional as F
import logging

# Добавляем корень проекта в пути
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from b_scr import ba_config
from b_scr.bj_models_dl import TextBiLSTM
from b_scr.bc_preprocessing import preprocess_text

# Настраиваем логирование
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def load_model_and_vocab(model_path, vocab_path, device='cpu'):
    """
    Загружает обученную модель и словарь из файлов.
    """
    # Загружаем словарь и маппинг классов
    vocab_data = torch.load(vocab_path, map_location=device)
    word2idx = vocab_data['word2idx']
    idx2label = vocab_data['idx2label']
    num_classes = len(idx2label)
    
    logger.info(f" Словарь загружен: {len(word2idx)}")
    logger.info(f" Классы: {list(idx2label.values())}")
    
    # Инициализируем модель с теми же параметрами, что и при обучении
    model = TextBiLSTM(
        vocab_size=len(word2idx),
        embedding_dim=ba_config.EMBEDDING_DIM,
        hidden_dim=ba_config.DL_HIDDEN_DIM,
        num_layers=ba_config.DL_NUM_LAYERS,
        num_classes=num_classes,
        dropout_rate=ba_config.DL_DROPOUT # При инференсе Dropout отключен
    ).to(device)
    
    # Загружаем обученные веса
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Переключаем в режим оценки
    
    logger.info(f" Модель загружена и готова к инференсу")
    
    return model, word2idx, idx2label

def predict_single_text(model, text, word2idx, idx2label, device='cpu'):
    """
    Делает предсказание для одного текста.
    """
    # 1. Очистка текста
    clean_text = preprocess_text(text)
    if not clean_text.strip():
        return "Неизвестно (текст полностью состоит из стоп-слов)", 0.0
    
    # 2. Токенизация и превращение в индексы
    words = clean_text.split()
    indices = []
    # Переводим каждое слово в число
    for word in words:
        if word in word2idx:
            indices.append(word2idx[word])
        else:
            indices.append(ba_config.UNK_IDX) # Неизвестное слово - <UNK>
    
    # 3. Обрезка до max_len
    if len(indices) > ba_config.MAX_SEQ_LENGTH:
        indices = indices[:ba_config.MAX_SEQ_LENGTH]
        
    # 4. Паддинг до max_len
    while len(indices) < ba_config.MAX_SEQ_LENGTH:
        indices.append(ba_config.PAD_IDX)
        
    # 5. Превращаем в тензор PyTorch
    input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
    
    # 6. Делаем предсказание
    with torch.inference_mode():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
        
    # 7. Находим класс с максимальной вероятностью
    pred_idx = probabilities.argmax(dim=1).item()
    confidence = probabilities[0, pred_idx].item()
    
    predicted_label = idx2label[pred_idx]
    
    return predicted_label, confidence

def main():
    logger.info(" Инференс Recurrent neural network (RNN)")
    
    # Определяем устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Используем устройство: {device}")
    
    # Пути к артефактам
    model_path = os.path.join(ba_config.MODELS_PATH, "bilstm_model.pt")
    vocab_path = os.path.join(ba_config.MODELS_PATH, "bilstm_vocabulary.pt")
    
    # Проверяем наличие файлов
    if not os.path.exists(model_path) or not os.path.exists(vocab_path):
        logger.error(" Модель или словарь не найдены! Сначала обучите модель.")
        return
    
    # Загружаем модель и словарь
    logger.info("\n Загрузка модели и словаря")
    model, word2idx, idx2label = load_model_and_vocab(model_path, vocab_path, device)
    
    # Тестовые тексты
    test_texts = [
        "Футбольный матч завершился победой сборной России со счетом 3:1",
        "Теннисист выиграл турнир Большого шлема в упорной борьбе",
        "Новый болид Формулы-1 показал рекордную скорость на трассе",
        "Лыжник занял первое место на зимних соревнованиях в Сочи",
        "Вчера играл в настольную игру с друзьями, было весело",
        "Вчера сделали с друзьями заказ в этом кафе, было весело, еда вкусная, обслуживание хорошее",
        "33 33 33"
    ]
    
    # Делаем предсказания
    logger.info("\n Тестирование модели")
    
    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n - Текст {i}: \"{text[:60]}{'...' if len(text) > 60 else ''}\"")
        
        predicted_label, confidence = predict_single_text(
            model, text, word2idx, idx2label, device
        )
        
        logger.info(f"  - Категория: {predicted_label}")
        logger.info(f"  - Уверенность: {confidence:.2%}")
        
    logger.info("\n Инференс завершен!")
    
if __name__ == "__main__":
    main()
    
        
    



