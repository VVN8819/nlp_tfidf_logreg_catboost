# nlp_tfidf_logreg_catboost

![Status](https://img.shields.io/badge/Status-Work%20in%20Progress-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge)

## Roadmap (План разработки)

- [x] Project structure creating
- [x] Raw data uploading and validation (`bb_data_loader.py`)
- [x] Exploratory Data Analysis (EDA)
- [ ] Text pre-processing (`preprocessing.py`)
- [ ] Features creation (TF-IDF)
- [ ] Model training and comparison (Logistic Regression vs CatBoost)
- [ ] Inference (`run_inference.py`)
- [ ] Final documentation and refactoring

# Sports Text Classifier 

Классификатор спортивных новостей на 4 категории: **winter_sport**, **football**, **tennis**, **autosport**.

## EDA
### Анализ целевой переменной (Классы)
1. Идеальный баланс - Все 13 категорий имеют примерно одинаковое количество постов (1962 - 2412). Это значит, что нам не придется бороться с дисбалансом классов при обучении моделей.
2. Дубликаты удалены: 2966 дубликатов (почти 8% данных) - это хороший результат очистки.

![Распределение категорий в обучающей выборке](image.png)

### Статистика длины текста
Сильное правостороннее смещение:
1. Большинство текстов короткие (0-100 слов)
2. Есть длинный хвост с редкими длинными текстами (до 504 слов)
3. Это типично для соцсетей: много коротких постов, мало длинных статей
Ключевые цифры:
1. min = 4 слова - это очень мало. Тексты из 4 слов почти бесполезны для классификации
2. median = 35 слов - половина всех текстов короче 35 слов
3. mean = 61 слово - средняя длина (выше медианы из-за длинных текстов)
4. max = 504 слова - есть полноценные статьи

![Статистика длины текста](image-1.png)

### Количество текстов длиной <= 10 слов: 2484
2484 текста (8.7% от всех) - это немного. И главное: они осмысленные.
Примеры:
- "Пожалуй празднования гола лучше в этом сезоне уже не будет"
- "Даниил Медведев передает привет из солнечного Монако"
- "Фабрицио Романо сообщает что Тен Хаг подписал контракт с МЮ"
Короткие тексты не нужно удалять. Они несут информацию и модель может их классифицировать.

### Частотный анализ (Поиск мусора и паттернов)
Анализ Топ-20 слов и Топ-15 слов категории показал:
- Много стоп-слов - решим с помощью NLTK
- 33 - артефакт удалим
- есть одиночные цифры не несут смысла - удалим

![Топ-20 слов](image-2.png)
![Топ-15 слов категории](image-3.png)

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
    ProcessedData[("a_data/ab_processed/preprocessed.csv")]

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

