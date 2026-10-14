# bc_preprocessing.py

"""
Модуль для предобработки (очистки и лемматизации) текста.
Правила очистки основаны на EDA датасета Russian Social Media Text Classification.
"""

import re
import spacy
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import logging

logger = logging.getLogger(__name__)

# 1. Инициализация инструментов
try:
    nlp = spacy.load('ru_core_news_sm')
except OSError:
    logger.error("Модель spaCy не найдена! Запустите: python -m spacy download ru_core_news_sm")
    raise

# Базовые стоп-слова NLTK
stop_words = set(stopwords.words('russian'))

# Расширяем стоп-слова на основе EDA (эти слова часто встречаются, но не несут смысла для классификации)
additional_stop_words = {
    'я', 'мы', 'все', 'это', 'тот', 'та', 'так', 'же', 'бы', 'ли',
    'свой', 'наш', 'ваш', 'его', 'её', 'их', 'мой', 'твой',
    'очень', 'хороший', 'новый', 'первый', 'последний', 'большой',
    'стать', 'быть', 'являться',
    'день', 'время', 'место', 'год', 'раз', 'часть', 'сторона',
    'весь', 'мочь', 'ещё', 'хотеть', 'сегодня', 'друг'
}
stop_words.update(additional_stop_words)

# Токенайзер: оставляет только буквы и цифры
nltk_tokenizer = RegexpTokenizer(r'\w+')

# Кэш для лемматизации
cache_dict = {}

# 2. Функции обработки
def get_norm_form(word: str) -> str:
    """Возвращает лемму слова с использованием кэша."""
    if word in cache_dict:
        return cache_dict[word]
    
    doc = nlp(word)
    norm_form = doc[0].lemma_
    cache_dict[word] = norm_form
    return norm_form

def preprocess_text(text: str) -> str:
    """
    Очищает текст из соцсетей на основе результатов EDA.
    
    Правила очистки:
    1. Нижний регистр
    2. Удаление артефакта "33" (15850 вхождений в датасете)
    3. Удаление одиночных цифр (1, 2, 3 и т.д.)
    4. Удаление слов с "tokenoid" (артефакт токенизации)
    5. Удаление ссылок (http, https)
    6. Токенизация (только буквы и цифры)
    7. Лемматизация
    8. Удаление стоп-слов (NLTK + дополнительные)
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Нижний регистр
    text = text.lower()
    
    # 2. Удаление артефакта "33"
    text = re.sub(r'\b33\b', '', text)
    
    # 3. Удаление одиночных цифр (1, 2, 3, ..., 9)
    text = re.sub(r'\b\d\b', '', text)
    
    # 4. Удаление слов с "tokenoid"
    text = re.sub(r'\w*tokenoid\w*', '', text)
    
    # 5. Удаление ссылок
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    
    # 6. Токенизация
    tokens = nltk_tokenizer.tokenize(text)
    
    # 7. Лемматизация
    tokens = [get_norm_form(token) for token in tokens]
    
    # 8. Удаление стоп-слов
    tokens = [token for token in tokens if token not in stop_words]
    
    # 9. Удаление слишком коротких токенов (1-2 символа после очистки)
    tokens = [token for token in tokens if len(token) > 2]
    
    # 10. Сборка
    return ' '.join(tokens)
    
    
    


