# nlp_tfidf_logreg_catboost

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)

## Roadmap (План разработки)

- [x] Project structure creating
- [x] Raw data uploading and validation (`bb_data_loader.py`)
- [x] Exploratory Data Analysis (EDA)
- [x] Text pre-processing (`bc_preprocessing.py`)
- [ ] Features creation (TF-IDF)
- [ ] Model training and comparison (Logistic Regression vs CatBoost)
- [ ] Inference (`run_inference.py`)
- [ ] Final documentation and refactoring

# Sports Text Classifier 

Sports news classifier with four categories: **winter_sport**, **football**, **tennis**, **autosport**.

## EDA
### Analysis of the target variable (Classes)
1. Perfect balance – All 13 categories have roughly the same number of posts (1,962–2,412). This means we won't have to deal with class imbalance when training the models.
2. Duplicates removed: 2,966 duplicates (almost 8% of the data) – a good cleaning result.

![Распределение категорий в обучающей выборке](plots/image.png)

### Text length statistics
Strong right skew:
1. Most texts are short (0–100 words)
2. There is a long tail of infrequent long texts (up to 504 words)
3. This is typical for social media: many short posts, few long articles
Key figures:
1. min = 4 words – very low; 4-word texts are almost useless for classification
2. median = 35 words – half of all texts are shorter than 35 words
3. mean = 61 words – average length (higher than the median due to long texts)
4. max = 504 words – includes full-length articles

![Статистика длины текста](plots/image-1.png)

### Number of texts with a length of ≤ 10 words: 2,484
2,484 texts (8.7% of the total) is not a large number. And most importantly: they are meaningful.
Examples:
- "Пожалуй празднования гола лучше в этом сезоне уже не будет"
- "Даниил Медведев передает привет из солнечного Монако"
- "Фабрицио Романо сообщает что Тен Хаг подписал контракт с МЮ"
Short texts do not need to be deleted. They contain information, and the model can classify them.

### Frequency analysis (searching for noise and patterns)
Analysis of the top 20 words and the top 15 category words revealed:
- Many stop words – we will handle this using NLTK
- "33" is an artifact – we will remove it
- There are standalone digits that carry no meaning – we will remove them

![Топ-20 слов](plots/image-2.png)
![Топ-15 слов категории](plots/image-3.png)

### Частотный анализ и Распределение длины текстов ПОСЛЕ очистки и лемматизации
Слова-паразиты (нужно добавить в стоп-слова):
- Местоимения и частицы
    'я', 'мы', 'все', 'это', 'тот', 'та', 'так', 'же', 'бы', 'ли',
    'свой', 'наш', 'ваш', 'его', 'её', 'их', 'мой', 'твой',
- Наречия и прилагательные (слишком общие)
    'очень', 'хороший', 'новый', 'первый', 'последний', 'большой',
- Глаголы-связки
    'стать', 'быть', 'являться',
- Общие существительные
    'день', 'время', 'место', 'год', 'раз', 'часть', 'сторона'

![Топ-20 слов ПОСЛЕ очистки и лемматизации](plots/image-4.png)
![Статистика длины текста ПОСЛЕ очистки и лемматизации](plots/image-5.png)

### Финальная очистка
- Слова "свой", "наш", "хороший", "стать", "первый", "быть", "новый" исчезли из топа
- Средняя длина текста уменьшилась с 38.68 до 36.89 слов
- В топе теперь больше спортивных терминов: "команда", "игра", "матч", "сезон", "турнир", "чемпионат", "игрок", "гонка", "победа", "чемпион"

## Быстрый старт

```bash
# Клонировать репозиторий
git clone <repo-url>
cd sports-classifier-tfidf

# Создать виртуальное окружение
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Установить зависимости
pip install -r requirements.txt

# Запустить обучение
python scripts/run_training.py

# Сделать предсказание
python scripts/run_inference.py --text "Футбольный матч завершился победой сборной России"
```

## Результаты

## Архитектура
```mermaid
graph TD
    %% Узлы: Данные
    RawData[("a_data/aa_raw/train.csv")]
    ProcessedData[("a_data/ab_processed/preprocessed_train.csv")]

    %% Узлы: Исходный код (src)
    DataLoader["b_src/bb_data_loader.py"]
    Preprocessing["b_src/preprocessing.py"]
    FeatureExtraction["b_src/feature_extraction.py"]
    ModelsTrain["b_src/models.py / train.py"]
    Evaluate["b_src/evaluate.py"]
    Predict["b_src/predict.py"]

    %% Узлы: Артефакты и Скрипты
    SavedModels[("models/vectorizer.pkl, model.cbm")]
    RunTrain["scripts/run_training.py"]
    RunInference["scripts/run_inference.py"]

    %% Поток данных обучения
    RawData -->|1. Загрузка сырых данных| DataLoader
    DataLoader -->|2. Передача DataFrame| Preprocessing
    Preprocessing -->|3. Сохранение чистого текста| ProcessedData
    ProcessedData -->|4. Чтение текста| FeatureExtraction
    FeatureExtraction -->|5. Создание TF-IDF матрицы| ModelsTrain
    ModelsTrain -->|6. Обучение и сохранение| SavedModels
    ModelsTrain -->|7. Расчет метрик| Evaluate

    %% Поток инференса
    RunTrain -.->|Запускает весь пайплайн| DataLoader
    RunInference -.->|Принимает новый текст| Predict
    SavedModels -.->|8. Загрузка весов модели| Predict
    Predict -->|9. Возврат: Категория + Уверенность| RunInference
```

## Структура проекта


## Примеры использования

## Технологии
- 'Python 3.10+'
- 'scikit-learn', 'CatBoost'
- 'NLTK', 'spaCy'
- 'pandas', 'numpy'

