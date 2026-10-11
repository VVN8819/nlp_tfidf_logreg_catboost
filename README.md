# nlp_tfidf_logreg_catboost

# Sports Text Classifier 

Классификатор спортивных новостей на 4 категории: **winter_sport**, **football**, **tennis**, **autosport**.

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
    RawData[("data/raw/train.csv")]
    ProcessedData[("data/processed/preprocessed.csv")]

    %% Узлы: Исходный код (src)
    DataLoader["src/data_loader.py"]
    Preprocessing["src/preprocessing.py"]
    FeatureExtraction["src/feature_extraction.py"]
    ModelsTrain["src/models.py / train.py"]
    Evaluate["src/evaluate.py"]
    Predict["src/predict.py"]

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

