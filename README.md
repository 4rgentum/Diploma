# Методика обнаружения несанкционированных действий и аномальной активности в корпоративной сети с использованием нейронных сетей

> Выпускная квалификационная работа (бакалавриат) — кафедра 42 НИЯУ МИФИ, 2026.
> Тема: разработка методики обнаружения аномальной активности в корпоративной сети на основе нейронных сетей с конечно-автоматным агентом активного тестирования.

Репозиторий содержит **полный материал работы**: четыре этапных отчёта в Markdown, итоговый отчёт по ГОСТ 7.32-2017 в XeLaTeX, программную реализацию `diploma_nids` на Python с пакетом тестов и презентационные материалы для защиты.

---

## Краткое содержание

Разработана воспроизводимая методика NIDS (Network Intrusion Detection System), объединяющая три подсистемы в едином протоколе:

1. **Нейросетевой детектор** — гибридная архитектура **CNN-LSTM**, обрабатывающая последовательности агрегированных потоковых признаков в скользящих окнах ($W=32$, $S=8$).
2. **Конечно-автоматный агент активного тестирования** — программный «противник» из библиотеки 7 параметризованных шаблонов трафика, FSM-планировщика (11 состояний) и инжектора дрейфа распределений. Обеспечивает прозрачную ground-truth по состояниям, недоступную в статических датасетах.
3. **Подсистема мониторинга дрейфа** на статистических индексах **PSI, KL и MMD** с калибровкой порогов под целевую FPR.

Реализация — на отечественно-совместимом стеке (Python 3.10+, PyTorch, scikit-learn, FastAPI, Streamlit), без GPU. Все эксперименты воспроизводимы одной командой; конфигурация — декларативная (YAML); случайные состояния зафиксированы.

### Ключевые количественные результаты на UNSW-NB15

| Модель | Тип | F1 (max) | PR-AUC | ROC-AUC | MCC |
|---|---|---:|---:|---:|---:|
| **CNN-LSTM (proposed)** | DL | **0,9735** | **0,9945** | **0,9923** | **0,9181** |
| XGBoost | classical | 0,9651 | 0,9932 | 0,9870 | 0,8885 |
| Random Forest | classical | 0,9623 | 0,9922 | 0,9853 | 0,8786 |
| GRU | DL | 0,9601 | 0,9894 | 0,9785 | 0,8725 |
| MLP | DL | 0,9548 | 0,9878 | 0,9770 | 0,8548 |
| Transformer | DL | 0,9543 | 0,9829 | 0,9683 | 0,8519 |
| BiLSTM | DL | 0,9517 | 0,9868 | 0,9750 | 0,8482 |

Полная таблица всех 15 моделей × до 3 seed-ов (22 запуска) — в [Stage3/results/tables/E1_summary.csv](Stage3/results/tables/E1_summary.csv) и [Stage4/results/tables/E1_summary.json](Stage4/results/tables/E1_summary.json).

После температурной калибровки (E3) ECE снижается с 0,089 до 0,047; рабочий порог под целевую FPR = 1 % достигается при $\tau = 0{,}4598$ с операционной полнотой 0,948.

---

## Структура репозитория

```
Diplomba/
├── ВКР.txt                       # Постановка задачи (тема, цель, актуальность, гипотеза, задачи, этапы)
├── README.md                     # Этот файл
├── .gitignore                    # Исключения для GitHub (датасеты, веса, кеши, локальные настройки)
│
├── Stage1/                       # Этап 1 — Аналитико-постановочный
│   ├── README.md
│   └── report_stage1.md          # Обзор подходов, сравнение датасетов, формализация требований
│
├── Stage2/                       # Этап 2 — Проектный
│   ├── README.md
│   └── report_stage2.md          # Детальное проектирование подсистем, протокол экспериментов
│
├── Stage3/                       # Этап 3 — Реализационный (Python-пакет diploma_nids)
│   ├── README.md
│   ├── report_stage3.md
│   ├── pyproject.toml
│   ├── requirements.txt
│   ├── Makefile
│   ├── configs/                  # YAML-конфигурации для всех подсистем
│   ├── src/diploma_nids/         # Исходный код пакета (8 подпакетов)
│   ├── scripts/                  # CLI-точки входа (download, train, eval, E1–E6, demo)
│   ├── tests/                    # pytest-тесты критических инвариантов
│   ├── experiments/runs/         # JSON-логи прогонов
│   └── results/                  # Таблицы и графики экспериментов
│
├── Stage4/                       # Этап 4 — Экспериментально-аналитический
│   ├── report_stage4.md          # Шесть экспериментов E1–E6 + cross-eval + demo
│   ├── experiments/              # Скрипты сборки сводных результатов и фигур
│   └── results/                  # Финальные таблицы, графики и runs
│
├── Report/                       # Итоговый отчёт по ГОСТ 7.32-2017
│   ├── report.tex                # Главный сборочный файл XeLaTeX
│   ├── preamble.tex              # Преамбула + biblatex-gost
│   ├── 1_title.tex … 12_appendix.tex
│   ├── bib/refs.bib              # 45 верифицированных источников
│   ├── figures/                  # Все рисунки итогового отчёта (PDF + PNG)
│   └── report.pdf                # Готовый PDF
│
├── build_presentation.py         # Сборка презентации защиты из шаблонов
├── Б22-505_Титов_НИР_Отчет.pdf  # Отчёт НИР для допуска
└── Б22_505_Титов_Нир_Презентация.{pdf,pptx}  # Презентация защиты
```

---

## Этапы работы

Работа выполнена в четыре этапа, **каждому соответствует свой отчёт** и раздел итогового документа в `Report/`.

### Этап 1. Аналитико-постановочный — [Stage1/](Stage1/)
Сравнительный анализ классических, ML- и DL-подходов к NIDS; обзор 9 нейросетевых архитектур; сравнение 10 публичных датасетов по 8 критериям (обоснован выбор UNSW-NB15); формализация 10 функциональных и 9 нефункциональных требований; три исследовательские гипотезы.

### Этап 2. Проектный — [Stage2/](Stage2/)
Детальное проектирование: CNN-LSTM как целевая архитектура (4,60/5 по 7 критериям сравнения); схема препроцессинга (cleanup → encoding → log/robust scaling → windowing); focal loss для балансировки классов; FSM-агент с 11 состояниями и библиотекой 7 шаблонов атак; drift-инжектор трёх типов; протокол шести экспериментов E1–E6.

### Этап 3. Реализационный — [Stage3/](Stage3/) (пакет `diploma_nids`)
Python-пакет с 8 подпакетами и 4230 строками кода: единый registry на 15 моделей (10 DL + 5 классических), training pipeline на focal loss / AdamW / cosine annealing / early stopping, evaluation pipeline с температурной калибровкой и bootstrap-CI 95 %, FSM-агент с runtime-режимами офлайн/стрим, FastAPI-сервис и Streamlit-дашборд для realtime-демо. 9 модулей pytest-тестов на критических инвариантах.

### Этап 4. Экспериментально-аналитический — [Stage4/](Stage4/)
Шесть экспериментов E1–E6 + cross-evaluation + демонстрационный прогон:

| № | Содержание |
|---|---|
| **E1** | Сравнение 15 моделей × до 3 seed-ов (22 запуска): F1, PR-AUC, ROC-AUC, MCC, FPR, latency |
| **E2** | Confusion matrix CNN-LSTM + per-attack recall по 10 классам атак UNSW-NB15 |
| **E3** | Температурная калибровка (ECE: 0,089 → 0,047) + threshold под target FPR = 0,01 |
| **E4** | Стресс-тест с FSM-агентом (600 тиков, 16 412 записей, per-state recall) |
| **E5** | Drift sweep: covariate × concept × 5 интенсивностей, 10 точек |
| **E6** | Latency для всех моделей, соответствие NFR-3 |
| **Cross-eval** | UNSW-NB15 → CICIDS2017 (оценка переносимости) |
| **Demo** | Realtime-pipeline с timeline-визуализацией и алертами |

---

## Установка и быстрый запуск

Требования: **Python 3.10+** (рабочая версия 3.13), любой современный CPU с AVX2, 8 ГБ RAM. GPU не нужен.

```powershell
# 1. Установить пакет (минимально)
cd Stage3
python -m pip install -e .

# Либо с веб-сервисом, дашбордом и тестами
python -m pip install -e ".[service,ui,dev]"

# 2. Скачать и распаковать UNSW-NB15 в data/unsw_nb15/
python scripts/01_download_unsw.py

# 3. Сборка обучающего набора + сериализация препроцессора
python scripts/02_build_dataset.py

# 4. Обучение CNN-LSTM (proposed)
python scripts/03_train.py `
    --model configs/models/cnn_lstm.yaml `
    --train configs/train/full.yaml `
    --seed 42

# 5. Оценка с калибровкой
python scripts/04_evaluate.py `
    --model configs/models/cnn_lstm.yaml `
    --checkpoint models/cnn_lstm_seed42.pt

# 6. Воспроизведение одного из шести экспериментов
python scripts/10_run_experiment_E1.py
```

### Realtime-сервис

```powershell
cd Stage3
python -m pip install -e ".[service]"

$env:DIPLOMA_MODEL_YAML="configs/models/cnn_lstm.yaml"
$env:DIPLOMA_CHECKPOINT="models/cnn_lstm_seed42.pt"
$env:DIPLOMA_PREPROCESSOR="data/processed/preprocessor.json"
$env:DIPLOMA_THRESHOLD="0.4598"

# FastAPI
python -m uvicorn diploma_nids.inference.service:create_app `
    --factory --host 127.0.0.1 --port 8000

# Streamlit-дашборд (в отдельном терминале)
python -m streamlit run src/diploma_nids/ui/streamlit_app.py
```

### Сборка итогового отчёта по ГОСТ

Нужен TeX-дистрибутив с XeLaTeX и Biber (TeX Live 2022+ или MiKTeX), пакеты `biblatex-gost`, `polyglossia`, `fontspec`, `lastpage`, системные шрифты Times New Roman / Arial / Courier New.

```powershell
cd Report
xelatex report.tex
biber   report
xelatex report.tex
xelatex report.tex

# либо одной командой
latexmk -xelatex -interaction=nonstopmode report.tex
```

---

## Технологический стек

| Слой | Инструменты |
|---|---|
| Язык | Python 3.10+ (CPython 3.13) |
| DL-фреймворк | PyTorch 2.11 (CPU-сборка) |
| Классические модели | scikit-learn 1.8, XGBoost 3.2 |
| Конфигурация / валидация | Pydantic 2.13, PyYAML |
| REST-сервис | FastAPI + Uvicorn |
| Визуализация | Streamlit, Matplotlib |
| Тестирование | pytest |
| Сборка отчёта | XeLaTeX + Biber + biblatex-gost (ГОСТ Р 7.0.100-2018) |

---

## Подсистема `diploma_nids`

Состав пакета (`Stage3/src/diploma_nids/`):

| Подпакет | Назначение | Ключевые модули |
|---|---|---|
| `utils` | I/O, seed, logging | `seed.py`, `io.py`, `logging.py` |
| `data` | Схема, загрузка, препроцессинг, окна, сплиты | `schema.py`, `preprocess.py`, `windowing.py`, `splits.py` |
| `models` | Единый registry на 15 моделей | `cnn_lstm.py`, `rnn_family.py`, `tcn.py`, `transformer.py`, `cnn1d.py`, `mlp.py`, `autoencoder.py`, `classical.py` |
| `training` | Loss + тренер | `losses.py` (Focal/BCE), `trainer.py` |
| `eval` | Метрики, калибровка, дрейф, error analysis | `metrics.py`, `thresholding.py`, `drift.py`, `error_analysis.py` |
| `attacker` | FSM-агент: шаблоны + планировщик + drift | `templates.py`, `agent.py`, `drift_injector.py`, `runtime.py` |
| `inference` | Online-инференс, алертинг, REST | `stream.py`, `alerting.py`, `service.py` |
| `ui` | Streamlit-дашборд | `streamlit_app.py` |

---

## Данные, веса и большие артефакты

В репозиторий **не попадают** (см. [.gitignore](.gitignore)):

- **Сырые датасеты** UNSW-NB15 и CICIDS2017 — распространяются авторами по отдельным лицензиям, скачиваются скриптом [Stage3/scripts/01_download_unsw.py](Stage3/scripts/01_download_unsw.py).
- **Обработанные выборки** в формате `.npz` — собираются скриптом [Stage3/scripts/02_build_dataset.py](Stage3/scripts/02_build_dataset.py).
- **Веса обученных моделей** (`*.pt`, `*.joblib`) — воспроизводятся скриптом [Stage3/scripts/03_train.py](Stage3/scripts/03_train.py) при фиксированных seed-ах.
- **LaTeX-промежутки** (`*.aux`, `*.bbl`, `*.log` и т. д.) — генерируются при каждой сборке.

В репозитории сохраняются: исходный код, конфигурации, тесты, JSON-логи прогонов (`experiments/runs/`), сводные таблицы, графики и итоговый PDF.

---

## Документы и итоговые материалы

| Документ | Файл |
|---|---|
| Постановка задачи ВКР | [ВКР.txt](ВКР.txt) |
| Отчёт Этапа 1 (аналитика) | [Stage1/report_stage1.md](Stage1/report_stage1.md) |
| Отчёт Этапа 2 (проектирование) | [Stage2/report_stage2.md](Stage2/report_stage2.md) |
| Отчёт Этапа 3 (реализация) | [Stage3/report_stage3.md](Stage3/report_stage3.md) |
| Отчёт Этапа 4 (эксперименты) | [Stage4/report_stage4.md](Stage4/report_stage4.md) |
| Итоговый отчёт по ГОСТ 7.32-2017 | [Report/report.pdf](Report/report.pdf) |
| Отчёт НИР (допуск) | [Б22-505_Титов_НИР_Отчет.pdf](Б22-505_Титов_НИР_Отчет.pdf) |
| Презентация защиты | [Б22_505_Титов_Нир_Презентация.pdf](Б22_505_Титов_Нир_Презентация.pdf) |

---

## Воспроизводимость

- Все случайные состояния зафиксированы; ведущие модели (CNN-LSTM, XGBoost, Random Forest, Logistic Regression) обучены на 3 seed-ах (42, 123, 2024) — для CNN-LSTM применяется усреднение вероятностей.
- Каждый прогон записывается в `Stage3/experiments/runs/{model}_seed{N}_{stage}.json` с полным составом метрик и параметров.
- Каждая фигура итогового отчёта пересобирается из соответствующего скрипта в `Stage4/experiments/rebuild_figures.py`.
- Конфигурации препроцессинга, обучения и оценки — YAML в `Stage3/configs/`, под ними хранится сериализованный `preprocessor.json`.

---

## Соответствие требованиям

- **ГОСТ 7.32-2017** — оформление отчёта: A4, поля 30/15/20/20 мм, Times New Roman 14 pt, междустрочный интервал 1,5, абзац 1,25 см, подписи «Рисунок N — Название» и «Таблица N — Название», сквозная нумерация в разделах.
- **ГОСТ Р 7.0.100-2018** — список источников через `biblatex-gost` стилем `gost-numeric`.
- **NFR-3** (эксплуатационная производительность) — латентность инференса CNN-LSTM не превышает целевое значение; результаты — [Stage3/results/tables/E6_latency.csv](Stage3/results/tables/E6_latency.csv).
- **FR-1…FR-10** и **NFR-1…NFR-9** — таблица соответствия в Этапе 1 и итоговом отчёте.

---

## Автор

**Титов** — студент группы Б22-505, кафедра 42 НИЯУ МИФИ, 2026.
Научный руководитель — см. титульный лист [Report/1_title.tex](Report/1_title.tex).
