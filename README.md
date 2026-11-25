# TirekAI — Оценка упражнения на разгибание локтя (E02)

Локальный воспроизводимый ML-пайплайн для домашней постинсультной реабилитации верхних конечностей (**Al-Sana Shakarim Solveathon — кейс TirekAI**). Система обрабатывает видео с веб-камеры или смартфона без платных облачных API, извлекает биомеханику локтевого сустава через **MediaPipe + OpenCV**, сегментирует повторения и возвращает калиброванную оценку качества выполнения каждого цикла.

---

## Архитектура решения

```text
TirekAI/
├── src/
│   ├── biomechanics.py      # Ядро MediaPipe (PoseLandmarker + ImageSegmenter fallback), фильтрация и find_peaks
│   ├── extract_features.py  # Шаг 1: сканирование датасета E02, извлечение временных рядов и признаков циклов
│   ├── model_bundle.py      # Сериализуемый ансамбль: StandardScaler + модель + калибровка вероятностей
│   ├── train.py             # Шаг 2: GroupShuffleSplit / StratifiedGroupKFold по subject_id, Baseline + CatBoost
│   └── inference.py         # Шаг 3: единая функция predict_video(video_path) для FastAPI / Streamlit
├── data/processed/
│   ├── e02_features.csv     # Признаки циклов + ресэмплированные траектории (T=64 и фаза разгибания T=16)
│   └── e02_timeseries.json  # Сглаженные кривые углов θ(t) по видео
├── artifacts/
│   ├── best_model.joblib    # Лучшая обученная модель с калиброванными вероятностями
│   ├── baseline_model.joblib# Интерпретируемый Baseline (LogisticRegression)
│   ├── scaler.joblib        # StandardScaler, обученный строго на train
│   ├── config.json          # Конфигурация признаков, CONFIDENCE_THRESHOLD и UI-статусов
│   ├── metrics.json         # Метрики на test и Leave-One-Volunteer-Out CV
│   ├── confusion_matrix.png # Сравнение матриц ошибок Baseline vs CatBoost
│   └── sample_angle_curve.png # Визуализация кривых угла θ(t) и найденных пиков разгибания
├── models/                  # Локальные веса MediaPipe (.task / .tflite), работающие офлайн и с кириллицей в путях
└── requirements.txt
```

### Ключевые инженерные решения
1. **Устойчивость к ракурсу и кириллическим путям Windows:**
   - Модели MediaPipe загружаются через `BaseOptions(model_asset_buffer=...)` напрямую из памяти Python, что исключает ошибку C++ `FileNotFoundError` при наличии кириллицы в путях (`Рабочий стол\...`).
   - Основной контур `PoseLandmarker` извлекает плечо (`11/12`), локоть (`13/14`), запястье (`15/16`) и таз (`23/24`) с фильтрацией по `visibility` и обработкой бокового ракурса; для приватных кропнутых записей крупного плана руки без корпуса в кадре автоматически включается нейросетевой контур `ImageSegmenter` с согласованной шкалой суставного угла.
2. **Гигиена данных и защита от утечек (`Data Leakage`):**
   - Автоматически обрабатывается расхождение имён файлов добровольца `04` в исходном датасете (`Complete/Train/04_02_00_*.mp4`).
   - Разбиение `train / val / test` выполняется строго по `subject_id` (`GroupShuffleSplit`) с жёсткими `assert`-проверками непересечения добровольцев + дополнительным аудитом `StratifiedGroupKFold` по всем добровольцам.
3. **Безопасная неклиническая интерпретация (`UI Status`):**
   - Модель возвращает калиброванную вероятность $p \in [0, 1]$. Если предсказанный класс `incomplete` **или** уверенность ниже `CONFIDENCE_THRESHOLD`, возвращается нейтральный статус **«Проверьте инструкцию»** (без медицинских диагнозов и рекомендаций).

---

## Быстрый старт (локальное развёртывание)

```powershell
# 1. Создание виртуального окружения (Python 3.10–3.12) и установка зависимостей
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt

# 2. Шаг 1: Извлечение признаков из видео упражнения E02
.\.venv\Scripts\python src/extract_features.py --data-dir "видео/TirekAI-dataset" --workers 6

# 3. Шаг 2: Обучение Baseline + CatBoost, подбор порога уверенности и генерация графиков
.\.venv\Scripts\python src/train.py

# 4. Шаг 3: Локальный инференс на любом видеофайле
.\.venv\Scripts\python src/inference.py "видео/TirekAI-dataset/An upper limb stroke rehabilitation exercise video/Exercise/2_Extending the Elbow/Complete/Test/08_02_01_01.mp4"
```

---

## Интеграция с FastAPI / Streamlit

```python
from src.inference import predict_video

result = predict_video("recorded_session.mp4")
# result["status"] -> "success" | "error"
# result["total_cycles"], result["completed_cycles"], result["review_cycles"]
# result["cycles"] -> список циклов с raw_class, ui_status, confidence, max_angle, min_angle
# result["angle_series"] -> покадровый ряд {"frame", "time_sec", "angle"} для графика на фронтенде
```
