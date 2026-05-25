# Stage 3 — Реализационный этап

Python-пакет `diploma_nids` — программная реализация методики из ВКР: нейросетевой детектор аномалий по потоковому трафику, конечно-автоматный агент активного тестирования и подсистема мониторинга дрейфа.

> Все архитектурные решения этого этапа опираются на результаты Stage 1 (выбор CNN-LSTM и UNSW-NB15 по сравнительному анализу) и проектные решения Stage 2 (детальная схема каждой подсистемы и протокол экспериментов E1–E6).

## Структура проекта

```
Stage3/
├── pyproject.toml                  # сборка пакета, опциональные группы (service/ui/dev)
├── requirements.txt                # ровно те же зависимости в плоском виде
├── Makefile                        # стандартные команды: install / test / train / experiments
├── configs/                        # YAML-конфигурации для каждой подсистемы
│   ├── data/                       # описание датасетов
│   ├── preprocess/                 # параметры Preprocessor + окон
│   ├── models/                     # 10 DL-моделей + бандл классических
│   ├── train/                      # сценарии обучения (default/full/smoke)
│   ├── eval/                       # пороги и параметры калибровки
│   ├── attacker/                   # FSM-политика, drift-конфиг
│   └── pipeline/                   # сборка realtime-демонстрации
├── src/diploma_nids/               # сам пакет (см. ниже)
├── scripts/                        # CLI-точки входа (download, train, eval, E1–E6, demo)
├── tests/                          # pytest-тесты критических инвариантов
├── data/                           # сырые и обработанные данные (не в git)
├── models/                         # обученные чекпойнты (не в git)
├── experiments/runs/               # JSON-логи прогонов
└── results/                        # таблицы и графики экспериментов
```

Состав `src/diploma_nids/`:

| Подпакет | Назначение | Ключевые модули |
|---|---|---|
| `utils` | I/O, seed, logging | `seed.py`, `io.py`, `logging.py` |
| `data` | Схема, загрузка, препроцессинг, окна, сплиты | `schema.py`, `load.py`, `preprocess.py`, `windowing.py`, `splits.py` |
| `models` | Единый регистр + 14 моделей | `base.py`, `cnn_lstm.py`, `rnn_family.py`, `tcn.py`, `transformer.py`, `cnn1d.py`, `mlp.py`, `autoencoder.py`, `classical.py` |
| `training` | Loss-функции + тренер | `losses.py` (Focal/BCE), `trainer.py` |
| `eval` | Метрики, калибровка, дрейф | `metrics.py`, `thresholding.py`, `drift.py`, `error_analysis.py` |
| `attacker` | Шаблоны + FSM + drift-инжектор | `templates.py`, `agent.py`, `drift_injector.py`, `runtime.py` |
| `inference` | Online-инференс, алертинг, REST | `stream.py`, `alerting.py`, `service.py` |
| `ui` | Streamlit-дэшборд | `streamlit_app.py` |

## Установка

```powershell
# Из корня репозитория
cd Stage3

# Минимальная установка
python -m pip install -e .

# С веб-сервисом + дашбордом + тестами
python -m pip install -e ".[service,ui,dev]"
```

Минимальные требования: **Python 3.10+**, актуально работает на 3.13. CPU-сборка PyTorch достаточна; GPU не требуется.

## Быстрый запуск

```powershell
# 1. Распаковать UNSW-NB15 в data/unsw_nb15 (можно автоматически)
python scripts/01_download_unsw.py

# 2. Сборка обучающего набора + сериализация препроцессора
python scripts/02_build_dataset.py

# 3. Обучение CNN-LSTM (proposed)
python scripts/03_train.py --model configs/models/cnn_lstm.yaml --train configs/train/full.yaml --seed 42

# 4. Оценка с калибровкой
python scripts/04_evaluate.py --model configs/models/cnn_lstm.yaml --checkpoint models/cnn_lstm_seed42.pt

# 5. Один из экспериментов
python scripts/10_run_experiment_E1.py
```

## Запуск тестов

```powershell
python -m pytest -q
```

## Запуск realtime-сервиса

```powershell
# 1. Установить пакет с extras "service"
python -m pip install -e ".[service]"

# 2. Запустить FastAPI
$env:DIPLOMA_MODEL_YAML="configs/models/cnn_lstm.yaml"
$env:DIPLOMA_CHECKPOINT="models/cnn_lstm_seed42.pt"
$env:DIPLOMA_PREPROCESSOR="data/processed/preprocessor.json"
$env:DIPLOMA_THRESHOLD="0.5"
python -m uvicorn diploma_nids.inference.service:create_app --factory --host 127.0.0.1 --port 8000

# 3. Дашборд (отдельный терминал)
python -m streamlit run src/diploma_nids/ui/streamlit_app.py
```

## Соответствие задачам ВКР

| Задача ВКР | Реализация |
|---|---|
| **4. Подсистема предобработки.** Схема, нормализация, кодирование, окна | `data/{schema,preprocess,windowing,load,splits}.py` |
| **5. Реализация и обучение модели.** Регистр моделей + Trainer | `models/`, `training/` |
| **6. Конечно-автоматный агент.** Шаблоны + FSM + drift-injector | `attacker/{templates,agent,drift_injector,runtime}.py` |
| **7. Прототип системы.** Online-инференс, REST, Streamlit, drift-monitor | `inference/`, `ui/`, `eval/drift.py` |

Отчёт по этапу — [report_stage3.md](report_stage3.md).
