# Этап 3. Реализационный

**Тема ВКР:** Методика обнаружения несанкционированных действий и аномальной активности в корпоративной сети с использованием нейронных сетей.

**Цель этапа:** полная программная реализация методики из Этапа 2 — подсистемы обработки данных, обучения и инференса 14 моделей (10 нейросетевых + 5 классических), конечно-автоматного агента активного тестирования с библиотекой шаблонов и инжектором дрейфа, прототипа онлайн-инференса с REST-интерфейсом и визуальной панелью.

**Покрываемые задачи ВКР:**

- задача 4 — программная реализация подсистемы предобработки и формирования выборок;
- задача 5 — программная реализация и обучение нейросетевых моделей в едином регистре;
- задача 6 — программная реализация конечно-автоматного агента активного тестирования;
- задача 7 — программная реализация прототипа системы в реальном времени.

**Связь со Stage 1 и Stage 2:**

- Все ключевые архитектурные решения опираются на сравнительный анализ Этапа 1: целевая архитектура — гибрид **CNN-LSTM** (4,60 баллов из 5 по семи критериям, см. [Stage1 §3.5](../Stage1/report_stage1.md#35-балльная-оценка-архитектур-по-критериям)); основной датасет — **UNSW-NB15** (4,80 баллов из 5 по восьми критериям, см. [Stage1 §4.5](../Stage1/report_stage1.md#45-взвешенная-балльная-оценка-и-решение-по-выбору)).
- Детальная схема каждой подсистемы и количественные параметры (W=32, S=8, AdamW, focal loss с α=0,25/γ=2,0, температурное масштабирование, PSI > 0,25 и т.д.) перенесены из [Stage 2](../Stage2/report_stage2.md) без отклонений.
- В реализации устранены проблемы, выявленные ревью первой версии прототипа: бо́льшая ёмкость целевой модели, эмпирические шаблоны атак вместо чисто параметрических (что устраняет OOD-смещение в E5), n ≥ 3 seed-ов из коробки для конкурирующих моделей.

---

## Оглавление

1. [Технологический стек и его обоснование](#1-технологический-стек-и-его-обоснование)
2. [Структура проекта](#2-структура-проекта)
3. [Подсистема обработки данных](#3-подсистема-обработки-данных)
4. [Реестр моделей и реализация 14 архитектур](#4-реестр-моделей-и-реализация-14-архитектур)
5. [Подсистема обучения](#5-подсистема-обучения)
6. [Подсистема оценки и калибровки](#6-подсистема-оценки-и-калибровки)
7. [Конечно-автоматный агент активного тестирования](#7-конечно-автоматный-агент-активного-тестирования)
8. [Подсистема инференса и алертирования](#8-подсистема-инференса-и-алертирования)
9. [REST-сервис и Streamlit-дэшборд](#9-rest-сервис-и-streamlit-дэшборд)
10. [Скрипты pipeline и экспериментов](#10-скрипты-pipeline-и-экспериментов)
11. [Тестовое покрытие](#11-тестовое-покрытие)
12. [Воспроизводимость и журналирование](#12-воспроизводимость-и-журналирование)
13. [Соответствие требованиям](#13-соответствие-требованиям)
14. [Выводы по этапу](#14-выводы-по-этапу)
15. [Источники](#15-источники)

---

## 1. Технологический стек и его обоснование

Стек выбран в Stage 2 §3.4 и подтверждён в Stage 3:

| Компонент | Назначение | Обоснование |
|---|---|---|
| **Python 3.10+** (3.13.11 в среде разработки) | основной язык | Поддерживается современный синтаксис аннотаций; на 3.13 проверено отсутствие конфликтов с зависимостями. |
| **PyTorch ≥ 2.2** (2.11 CPU в среде) | глубокое обучение | Эталонный фреймворк для DL; единственный устоявшийся с детерминированными CPU-ядрами. CPU-сборка удовлетворяет требованиям проекта (GPU не нужен) [[1]](#ref-pytorch). |
| **scikit-learn ≥ 1.4** | классические baselines | Зрелая библиотека с устоявшимся API `fit/predict/predict_proba`; реализации LR, RF, IsolationForest, OCSVM [[2]](#ref-sklearn). |
| **XGBoost ≥ 2.0** | gradient boosting baseline | Лидер на табличных данных по широкому корпусу публикаций 2019–2025; реализация градиентного бустинга с режимом ``tree_method=hist`` для CPU-эффективности [[3]](#ref-xgboost). |
| **Pydantic v2** | валидация конфигов и схем | Установлен стандартом для валидации данных в Python; v2 даёт примерно 10x ускорение к v1 [[4]](#ref-pydantic). |
| **PyYAML 6** | декларативные конфигурации | Стандарт для YAML в Python; используется во всех конфигурациях пайплайна. |
| **FastAPI ≥ 0.110** (опционально) | REST-сервис инференса | Современный фреймворк с автоматической OpenAPI-документацией; устоявшийся выбор для production ML-сервисов [[5]](#ref-fastapi). |
| **Streamlit ≥ 1.32** (опционально) | UI-дэшборд | Минимальные накладные расходы для визуализации потоков; типовая практика для ML-демо. |
| **pytest 8+** | юнит-тестирование | Стандарт-фактор; параметризованные тесты используются для прогона smoke-тестов по всем 15 моделям. |
| **matplotlib ≥ 3.8** | графики экспериментов | Использован вместо seaborn для минимизации зависимостей; figures сохраняются и в PNG, и в PDF. |
| **NumPy ≥ 1.26 / pandas ≥ 2.2 / SciPy ≥ 1.13** | базовая математика и I/O | Без альтернатив. |

### Альтернативы и почему отклонены

- **TensorFlow / Keras** — отклонён в пользу PyTorch по 4 аргументам: (а) более устойчивое управление seed-ом и детерминированными ядрами на CPU, (б) проще динамические графы (важно для FSM-агента), (в) большая часть современных публикаций по NIDS 2024–2025 идёт на PyTorch [[6]](#ref-cnnlstm-ieee-2024), (г) меньший размер сборки.
- **JAX** — отклонён ввиду меньшей зрелости экосистемы для табличных задач и сложности интеграции с sklearn / XGBoost.
- **LightGBM / CatBoost** — XGBoost выбран как наиболее воспроизводимый и широко цитируемый в NIDS-литературе; LightGBM сохранён как возможное расширение.
- **Polars** — pandas сохранён как опорный для совместимости со sklearn / pydantic.
- **Flask / Django** — FastAPI предпочтён по причинам автоматической OpenAPI-документации и встроенной поддержки Pydantic-моделей [[5]](#ref-fastapi).
- **Plotly Dash** — Streamlit выбран как более лёгкий в развёртывании для демо-сценария.

Все версии зафиксированы в [pyproject.toml](pyproject.toml) и [requirements.txt](requirements.txt) как минимальные нижние границы. Для CI / production имеет смысл зафиксировать exact-pin при выпуске.

---

## 2. Структура проекта

Полная структура — в [README.md](README.md). Краткая схема:

```
Stage3/
├── pyproject.toml / requirements.txt / Makefile / .gitignore
├── configs/                # YAML-конфигурации (data, preprocess, models, train, eval, attacker, pipeline)
├── src/diploma_nids/
│   ├── utils/              # seed, io, logging
│   ├── data/               # schema, load, preprocess, windowing, splits
│   ├── models/             # 14 моделей в едином registry
│   ├── training/           # trainer + focal/BCE loss
│   ├── eval/               # metrics, thresholding, drift, error_analysis
│   ├── attacker/           # templates, FSM, drift injector, runtime
│   ├── inference/          # alerting, stream, FastAPI service
│   └── ui/                 # Streamlit dashboard
├── scripts/                # CLI: 01–07 (pipeline), 10–15 (E1–E6), 30 (demo)
└── tests/                  # 86 pytest-тестов
```

Общий объём исходного кода — порядка 4 000 строк Python, без учёта тестов и конфигов.

---

## 3. Подсистема обработки данных

Реализация в [`src/diploma_nids/data/`](src/diploma_nids/data/).

### 3.1. Схема и загрузка

`data/schema.py` — единственный источник истины: 44 признака UNSW-NB15 разделены на пять групп (basic / content / time / additional / ct_*), плюс категориальные ``proto, service, state``, бинарные ``is_ftp_login, is_sm_ips_ports`` и список ``UNSW_LOG_TRANSFORM`` для тяжёлохвостых признаков (Stage 2 §2.1).

`data/load.py` реализует:

* `load_unsw_nb15(data_dir)` — читает официальные train/test CSV; принудительно приводит типы и канонизирует категориальные значения (`tcp` / `TCP` / ` tcp` → `tcp`).
* `load_cicids2017(data_dir)` — читает все per-day CSV из CICFlowMeter, переименовывает столбцы в UNSW-подобные имена для cross-evaluation (Stage 2 §8.10), бинаризует метку (BENIGN vs прочее).

### 3.2. Препроцессор

`data/preprocess.py` — sklearn-совместимый `Preprocessor.fit/transform/fit_transform/save/load` с **полной JSON-сериализацией** состояния (`PreprocessorState`). Решения проектирования:

* **Все статистики собираются только на train** — медианы и IQR робастного масштабирования, частоты для frequency-encoding, квантильные границы клиппинга и список one-hot-категорий фиксируются на `fit`. Это исключает test-time leakage по построению.
* **Двухступенчатая нормализация числовых:** ``log1p`` для тяжёлохвостых (Stage 2 §2.4) → **Robust Scaling** (median / IQR). Робастные оценки выбраны вместо StandardScaler / MinMax по причине устойчивости к выбросам (типичная практика для тяжёлохвостых распределений в NIDS) [[7]](#ref-scikit-robust).
* **Кодирование категориальных:** one-hot для малой кардинальности (`proto`, `state`), frequency encoding для большой (`service`). Неизвестная категория в эксплуатации обрабатывается единообразно (one-hot → indicator `unknown`, frequency → 0,0).
* **Сериализация** — JSON, без pickle, — это безопаснее и устойчивее к версии Python/sklearn.

### 3.3. Скользящие окна

`data/windowing.py` реализует векторизованный `build_windows(features, labels, window=32, stride=8, agg='last')` с тремя стратегиями агрегации меток (`last`, `majority`, `any`). Использован `np.lib.stride_tricks.sliding_window_view` — без копирования данных, что важно для потокового режима.

### 3.4. Разделение выборок

`data/splits.py` реализует:

* `train_val_split` — стратифицированное случайное разделение по метке;
* `temporal_split` — split по временному порядку (на случай уже отсортированного по времени capture).

---

## 4. Реестр моделей и реализация 14 архитектур

Реализация в [`src/diploma_nids/models/`](src/diploma_nids/models/).

### 4.1. Регистр и базовые абстракции

`models/base.py` определяет:

* `BaseModel` — общий протокол для всех моделей с методами `predict_proba`, `predict`, `output`. Возвращает `ModelOutput(logits, probs, extras)`.
* `BaseDeepModel(nn.Module, BaseModel)` — PyTorch-база, принимает вход `(B, W, F)`, возвращает logit `(B,)`. Метод `predict_proba` уже реализован на верхнем уровне через `torch.sigmoid(forward(x))`.
* Декоратор `@register("name", **meta)` — единая точка регистрации; `available_models()` возвращает справочник имя → метаданные; `build_model(cfg)` создаёт инстанс из YAML-конфига.

### 4.2. Реализованные модели

В реестре зарегистрированы **15 моделей** (10 DL + 5 классических — отличие от Stage 2: добавлены `vae` и `gru` как отдельные сущности, что даёт итого 15; в эксперименте E1 сравниваются все 15):

```
['autoencoder', 'bilstm', 'cnn1d', 'cnn_lstm', 'gru',
 'isolation_forest', 'logistic_regression', 'lstm', 'mlp', 'ocsvm',
 'random_forest', 'tcn', 'transformer', 'vae', 'xgboost']
```

| Модель | Файл | Краткое описание |
|---|---|---|
| **CNN-LSTM** (proposed) | `cnn_lstm.py` | Conv1D×2 → BiLSTM → attention pool → MLP head. Расширенный по сравнению с baseline-описанием Stage 2 для повышения F1: bidirectional + attention pool + LayerNorm. |
| MLP | `mlp.py` | Flatten + 2 FC + dropout. |
| 1D-CNN | `cnn1d.py` | Conv1D×2 + adaptive pool + FC. |
| LSTM / GRU / BiLSTM | `rnn_family.py` | Параметризованная семья; BiLSTM по умолчанию с attention. |
| TCN | `tcn.py` | Bai et al. 2018: причинные дилатированные свёртки с residual. |
| Transformer-encoder | `transformer.py` | Sinusoidal PE + 2 encoder-блока с mean pooling. |
| Autoencoder | `autoencoder.py` | MLP-энкодер/декодер; обучается на normal-only, аномальный балл — reconstruction error → logit. |
| VAE | `autoencoder.py` | Расширение AE с reparametrisation trick и KL-регуляризацией. |
| LR / RF / XGBoost | `classical.py` | sklearn / XGBoost через единый `_ClassicalAdapter` с встроенным StandardScaler для линейных моделей. |
| IsolationForest | `classical.py` | sklearn, обучается только на normal-сэмплах. |
| OCSVM | `classical.py` | sklearn, one-class режим. |

Все DL-модели принимают `(B, W, F)` и возвращают `(B,)` logit; классические — flattened `(B, W*F)` → `(B,)` proba. Это даёт **честное сравнение в одинаковых условиях входа** для эксперимента E1.

### 4.3. Усиления CNN-LSTM против первой версии

В ревью первой реализации CNN-LSTM показал F1 = 0,77 — далеко от NFR-1a (≥ 0,90). Текущая реализация содержит четыре улучшения, повышающих шансы достичь требования при той же длине окна:

1. **Bidirectional LSTM** — стандартный приём в публикациях CNN-LSTM на UNSW-NB15 2024–2025 [[6]](#ref-cnnlstm-ieee-2024).
2. **Attention pooling** над выходами LSTM вместо последнего скрытого состояния — стабилизирует обучение по seed.
3. **LayerNorm перед головой** — снимает scale drift между Conv и LSTM активациями.
4. **Расширенный MLP-head** (64-dim hidden вместо 32) — больше ёмкости для разделения тонкой структуры классов.

Финальное число параметров — около 200 тыс., размер модели — около 0,8 МБ.

---

## 5. Подсистема обучения

Реализация в [`src/diploma_nids/training/`](src/diploma_nids/training/).

### 5.1. Loss-функции

`training/losses.py`:

* **`FocalLoss`** (Lin et al. 2017 [[8]](#ref-focal-loss-2017)) — численно стабильная реализация на основе `binary_cross_entropy_with_logits`. Параметры по умолчанию α=0,25, γ=2,0 (рекомендация Stage 2 §2.6). Принимается как базовая loss-функция для DL-моделей.
* **`WeightedBCE`** — BCE с опциональным `pos_weight`.
* **`build_loss(cfg)`** — dispatch helper, читает YAML-секцию `loss`.

### 5.2. Trainer

`training/trainer.py` — единый `Trainer` для всех DL-моделей. Особенности:

* **TrainerConfig** (`dataclass`) — все гиперпараметры из Stage 2 Table 3.1 в одном объекте; конструируется из YAML.
* **AdamW + cosine annealing → 1e-6** — обоснование в Stage 2 §3.4 (Loshchilov & Hutter 2019, ICLR) [[9]](#ref-adamw-2017).
* **Early stopping по `val_pr_auc` с patience=7** — PR-AUC основная метрика при дисбалансе.
* **Gradient clipping** `‖g‖₂ ≤ 1,0` — защита от взрыва градиентов в LSTM.
* **Деflective AE/VAE** — при `is_unsupervised_ae=True` обучение идёт только на normal-сэмплах, лосс — MSE реконструкции (+ KL для VAE).
* **Reproducible** — `set_seed` вызывается до конструктора; `torch.use_deterministic_algorithms(True, warn_only=True)`.

История обучения возвращается как `TrainHistory` (`train_loss[]`, `val_loss[]`, `val_pr_auc[]`, `val_f1[]`, `lr[]`, `best_epoch`, `total_time_sec`, `early_stopped`) — это JSON-сериализуется и сохраняется в `experiments/runs/<model>_<seed>_train.json`.

### 5.3. Train helpers

* `train_deep(model, X_tr, y_tr, X_val, y_val, cfg, save_path)` — обучение DL-модели + сохранение чекпойнта.
* `train_classical(model, X_tr, y_tr, save_path)` — обучение классической модели через `_ClassicalAdapter.fit` + сохранение через `joblib`.

---

## 6. Подсистема оценки и калибровки

Реализация в [`src/diploma_nids/eval/`](src/diploma_nids/eval/).

### 6.1. Метрики

`eval/metrics.py`:

* `binary_metrics(y_true, y_score, threshold=0.5)` — Precision, Recall, F1, FPR, MCC, ROC-AUC, PR-AUC, tp/tn/fp/fn. Защищён от вырожденных случаев (single-class input).
* `expected_calibration_error(y_true, y_prob, n_bins=15)` — ECE с биннами равной ширины (Guo et al. 2017 [[10]](#ref-temperature-scaling-2017)).
* `bootstrap_ci(metric_fn, y_true, y_score, n_resamples=200, ci=0.95)` — bootstrap 95% CI.
* `fp_per_time(timestamps, is_fp, bin_seconds=60)` — для эксплуатационных метрик.
* `time_to_detect(timestamps, y_true, y_pred)` — медианное TTD по эпизодам атак.

### 6.2. Калибровка

`eval/thresholding.py`:

* **`TemperatureScaler`** — Guo et al. 2017 [[10]](#ref-temperature-scaling-2017). Однопараметрическая модель `T`, оптимизация через L-BFGS на валидационных логитах. Включён fallback на grid search при сбое L-BFGS на маленьких выборках.
* **`find_threshold_for_target_fpr(y_true, y_score, target_fpr=0.01)`** — O(N log N) через сортировку score-ов отрицательного класса. Возвращает наименьший порог, при котором FPR ≤ target.
* **`find_threshold_for_f1_max(y_true, y_score)`** — grid search по 201 точке.

### 6.3. Drift-monitor

`eval/drift.py`:

* `psi(ref, cur, n_bins=10)` — Population Stability Index с квантильными биннами (устойчивее к разной длине distributions). Порог 0,25 — индустриальный стандарт [[11]](#ref-fiddler-psi-2024).
* `kl_divergence(ref, cur, n_bins=20)` — KL-расхождение на гистограммах одинаковых биннов.
* `mmd_rbf(ref, cur, max_samples=1000)` — MMD² с гауссовым ядром, **bandwidth = медиана попарных дистанций** (медианная эвристика, Garreau et al. 2017) [[12]](#ref-mmd-median-2014). Подсэмплирование до 1000 точек ограничивает ``O(n²)`` вычисление.
* `drift_report(reference, current, psi_threshold=0.25)` — агрегированный отчёт с `drift_alarm = (mean_psi > threshold) OR (>=10% признаков превысили порог)`.

### 6.4. Анализ ошибок

`eval/error_analysis.py` — `per_class_recall(y_true, y_pred, attack_cat)` возвращает per-class recall (или FPR для класса Normal) в виде DataFrame.

---

## 7. Конечно-автоматный агент активного тестирования

Реализация в [`src/diploma_nids/attacker/`](src/diploma_nids/attacker/).

### 7.1. Шаблоны атак

`attacker/templates.py` — **ключевое изменение по сравнению с первой версией прототипа**: реализован **дуальный режим**:

1. **Empirical mode (по умолчанию в pipeline)** — `build_templates_from_dataframe(df)` строит per-feature sampler из эмпирических распределений соответствующего подмножества UNSW-NB15 (`attack_cat == <cat>`). Для числовых признаков — inverse-CDF sampling с малым гауссовым jitter; для категориальных — выбор по эмпирической MMF; для бинарных — Bernoulli с эмпирическим `p`. Результирующее распределение по построению близко к источнику → PSI вблизи нуля при intensity = 0.
2. **Parametric mode (fallback)** — `default_template_registry()` возвращает 8 вручную параметризованных шаблонов (`normal`, `ddos_synflood`, `port_scan`, `brute_force`, `exploit_payload`, `data_exfil`, `internal_recon`, `worm_lateral`) с семантически осмысленными диапазонами. Используется, когда датасет ещё не доступен.

Этот дуальный режим **закрывает корневую проблему предыдущей версии** (PSI ≈ 0,31 при intensity = 0 из-за OOD-смещения параметрических шаблонов).

### 7.2. FSM-агент

`attacker/agent.py`:

* **FSMConfig** — алфавит из 11 состояний (NORMAL + 7 ATTACK + 3 DRIFT), стохастическая матрица переходов, словарь минимальных дворов, map состояния → имя шаблона.
* **FSMAgent** — конструктор проверяет инвариант суммы вероятностей в каждой строке = 1,0 (проверяется тестом).
* **AgentTick** — структура эмиссии: `timestamp`, `flow`, `fsm_state`, `attack_cat`, `drift_type`, `intensity`. Это и есть прозрачная **ground-truth** для эксперимента E4 (per-state recall).

### 7.3. Drift-injector

`attacker/drift_injector.py` — три типа дрейфа:

* **Covariate** (`DRIFT_COV`) — `x' = (1 + α·u)·x + α·σ·ε`, где `u ~ U(-0.5, 0.5)`, `ε ~ N(0,1)`, `σ` — std признака на train.
* **Prior** — реализован на уровне FSM-политики (изменение долей классов), не на уровне отдельного flow.
* **Concept** — маскирование сигнатурных признаков атаки: случайное обнуление `ct_*` счётчиков, подмена `state` и `service` на нормальные значения. Применяется только к flow с `label=1`.

Интенсивность α контролируется параметром в (0, 1]; α=0 → дрейф не применяется (no-op).

### 7.4. Runtime

`attacker/runtime.py` — `AttackerRuntime` объединяет FSM-агента и (опционально) drift-injector. Два режима:

* `collect_dataframe(n_ticks)` — оффлайн: возвращает `pd.DataFrame` с дополнительными колонками `fsm_state`, `drift_type`, `intensity`, `timestamp` для E4/E5.
* `stream(n_ticks)` / `stream_with_callback(n_ticks, cb)` — потоковая выдача `AgentTick` объектов в детектор.

`runtime_from_config(policy_path, empirical_df=...)` — фабрика, читает YAML-политику и строит соответствующий runtime.

---

## 8. Подсистема инференса и алертирования

Реализация в [`src/diploma_nids/inference/`](src/diploma_nids/inference/).

### 8.1. WindowScorer

`inference/stream.py` — потоковый scorer:

* `push(row)` — принимает один FlowRecord, накапливает в очереди размера `window`, при готовности окна (с учётом stride) возвращает скоринг.
* `score_batch(df)` — оффлайн-режим: возвращает `(end_idx, score)` для DataFrame.

Использует тот же `Preprocessor` JSON, что был сохранён в обучении — нулевая разница между обучением и инференсом.

### 8.2. AlertFormer

`inference/alerting.py`:

* **`Severity`** — Enum с пятью уровнями (info / low / medium / high / critical).
* **`severity_from_ratio(score / τ)`** — детерминированная функция шкалирования (Stage 2 §5.2). Пороги: 1,1 → low, 1,3 → medium, 1,6 → high, 2,0 → critical.
* **`AlertFormer.maybe_emit(...)`** — основная логика: если score < τ, возвращает None; иначе пробует дедуплицировать с предыдущими алертами в окне `dedup_seconds` по триплету `(src_ip, dst_ip, attack_cat)`; при попадании в дедуп — обновляет счётчик и max-severity и возвращает None.
* История фиксированного размера (`deque(maxlen=N)`) — экспорт через `recent(n)`.

### 8.3. Формат алерта

JSON Lines, поля: `ts`, `score`, `threshold`, `severity`, `src_ip`, `dst_ip`, `attack_cat`, `dedup_count`, `model_version`, `ground_truth` (FSM-state и drift-type при тестировании), `extras`.

---

## 9. REST-сервис и Streamlit-дэшборд

### 9.1. FastAPI service

`inference/service.py` — фабрика `create_app()` (важно: фабрика, а не модуль-уровень app, чтобы поддерживать ленивую загрузку и переменные окружения). Конфигурация через env:

| Переменная | Назначение |
|---|---|
| `DIPLOMA_MODEL_YAML` | путь к YAML конфигу модели |
| `DIPLOMA_CHECKPOINT` | путь к чекпойнту (.pt или .joblib) |
| `DIPLOMA_PREPROCESSOR` | путь к JSON-препроцессору |
| `DIPLOMA_THRESHOLD` | калиброванный порог τ |
| `DIPLOMA_MODEL_VERSION` | произвольная строка-маркер версии модели |
| `DIPLOMA_WINDOW` / `DIPLOMA_STRIDE` | параметры WindowScorer |

Эндпоинты:

* `GET /health` — liveness.
* `GET /info` — порог, окно, версия модели, имена выходных признаков.
* `POST /score` — батч flow-записей; возвращает скоринги и алерты.
* `POST /score-window` — прямой скоринг готового окна `(W, F)`.
* `GET /alerts/recent?n=N` — последние N алертов.

### 9.2. Streamlit dashboard

`ui/streamlit_app.py` — обновляемая каждые 2 с панель: метрики (число алертов, доля critical+high, средний скоринг), таблица последних 50 алертов и временной ряд скорингов. Соединяется с FastAPI-сервисом через `DIPLOMA_API_URL`.

---

## 10. Скрипты pipeline и экспериментов

Все скрипты — самостоятельные CLI с `argparse` и читаемыми именами выходных файлов.

| Скрипт | Назначение |
|---|---|
| `01_download_unsw.py` | Распаковать `unsw.zip` в `data/unsw_nb15/`. |
| `02_build_dataset.py` | Загрузить UNSW, поделить train/val, обучить препроцессор, построить окна, сохранить `.npz`. |
| `03_train.py` | Обучить любую зарегистрированную модель из YAML-конфига для заданного `--seed`. |
| `04_evaluate.py` | Оценить чекпойнт на val/test, T-scaling, target-FPR порог, per-class recall. |
| `05_run_attacker.py` | Прогон FSM-агента N тиков, сохранение DataFrame. |
| `06_drift_test.py` | Smoke-проверка drift-monitor. |
| `07_demo_realtime.py` | Полный demo-прогон (FSM → детектор → алерт-формер → drift), генерация timeline PNG/PDF. |
| `10_run_experiment_E1.py` | E1 — сравнение 14 моделей в равных условиях, многих seed-ов. |
| `11_run_experiment_E2_error_analysis.py` | E2 — confusion matrix и per-class recall для CNN-LSTM. |
| `12_run_experiment_E3_calibration.py` | E3 — T-scaling + target-FPR порог для всех DL-моделей. |
| `13_run_experiment_E4_attacker.py` | E4 — стресс-тест с FSM-агентом, per-state recall. |
| `14_run_experiment_E5_drift.py` | E5 — drift sweep (covariate / concept × 5 intensities). |
| `15_run_experiment_E6_perf.py` | E6 — латентность и throughput на CPU. |

Каждый скрипт сохраняет результаты в `experiments/runs/*.json` (метрики) и `results/{tables, figures}/` (CSV + PNG/PDF).

---

## 11. Тестовое покрытие

Тесты — в [`tests/`](tests/). Запуск: `python -m pytest -q`. На текущей среде проходит **86 тестов**.

| Файл | Покрываемые инварианты |
|---|---|
| `test_utils.py` | reproducibility set_seed; YAML/JSON roundtrip; ensure_dir создаёт родительские директории; dump_json обрабатывает numpy. |
| `test_configs.py` | все YAML парсятся; сумма вероятностей FSM-переходов = 1,0; все ожидаемые модели имеют configs. |
| `test_preprocess.py` | fit→transform без NaN; save/load roundtrip; неизвестная категория обрабатывается; transform без fit падает; статистики не меняются при transform. |
| `test_windowing.py` | shapes по формуле; стратегия last; все 3 агрегации работают; короткий вход падает; невалидные параметры падают. |
| `test_splits.py` | split дисъюнктный и полный; reproducible; temporal сохраняет порядок. |
| `test_models_smoke.py` | все 15 моделей в реестре; forward у DL даёт `(B,)`; classical fit+predict выдаёт `[0,1]`; CNN-LSTM помечена как `proposed=True`. |
| `test_losses.py` | FocalLoss > 0; correct < wrong; build_loss dispatch. |
| `test_metrics.py` | perfect metrics; single-class не крашит; target-FPR threshold даёт FPR ≤ target; ECE не растёт после калибровки; bootstrap CI содержит точку. |
| `test_drift.py` | PSI ≈ 0 на одинаковых; PSI растёт со сдвигом; KL ≈ 0; MMD ≈ 0; drift_report срабатывает на сдвиге и молчит на идентичных. |
| `test_attacker.py` | FSM эмитит N тиков; invalid policy падает; runtime собирает DataFrame; drift_injector действительно меняет flow; empirical templates строятся; from_config работает. |
| `test_alerting.py` | severity ladder; алерт пропускается под порогом; дедуп склеивает; за окном дедуп выключается; max history; positive threshold required. |

Все тесты используют общий `conftest.py` с `toy_unsw_df` фикстурой (schema-compatible mini-DataFrame), что даёт быстрые smoke-проверки без зависимости от наличия CSV на диске.

---

## 12. Воспроизводимость и журналирование

* **Seed-инжиниринг.** `utils/seed.set_seed(s)` фиксирует `PYTHONHASHSEED`, `random`, `numpy`, `torch` (CPU+CUDA), включает `torch.use_deterministic_algorithms(True, warn_only=True)`. Это вызывается в первой строке каждого скрипта.
* **Артефакты обучения** — JSON в `experiments/runs/<model>_seed<S>_train.json`, чекпойнты в `models/<model>_seed<S>.{pt,joblib}`.
* **Декларативная параметризация** — все стадии настраиваются через YAML; команды CLI принимают `--seed` явно.
* **`make reproduce`** — `preprocess → train(cnn_lstm) → evaluate → all experiments` за одну команду.
* **Препроцессор сериализуется JSON** — побитово идентичный transform в обучении и инференсе.

---

## 13. Соответствие требованиям

Сопоставление с FR/NFR из [Stage 1 §6](../Stage1/report_stage1.md#6-формализация-функциональных-и-нефункциональных-требований-к-системе):

| ID | Требование | Реализация |
|---|---|---|
| **FR-1** | Поддержка схемы UNSW-NB15 | `data/schema.py` + валидация в `load_unsw_nb15` |
| **FR-2** | Препроцессинг + окна | `data/preprocess.py`, `data/windowing.py` |
| **FR-3** | ≥ 14 моделей в едином регистре | 15 моделей зарегистрированы, проверено тестом |
| **FR-4** | Калибровка вероятностей + порог | `eval/thresholding.TemperatureScaler` + `find_threshold_for_target_fpr` |
| **FR-5** | Алерты с severity и дедупликацией | `inference/alerting.AlertFormer` |
| **FR-6** | REST API | `inference/service.create_app()` (FastAPI, 5 эндпоинтов) |
| **FR-7** | Streamlit-дэшборд | `ui/streamlit_app.py` |
| **FR-8** | FSM-агент с ≥ 7 шаблонами + drift трёх типов | `attacker/{templates, agent, drift_injector, runtime}.py` |
| **FR-9** | Drift-monitor PSI/KL/MMD | `eval/drift.py` с `drift_report` |
| **FR-10** | Seed + журналы | `utils/seed.py` + JSON logs в `experiments/runs/` |

NFR оцениваются в Stage 4 (эксперименты). Инфраструктура для каждого NFR готова: NFR-1/1b (E1 + E3), NFR-2 (E3), NFR-3/4 (E6), NFR-5 (E4), NFR-6 (E5), NFR-7 (cross-eval), NFR-8 (`make reproduce`), NFR-9 (`--seed` в скриптах + `seeds_main` в E1).

---

## 14. Выводы по этапу

1. **Реализован полный программный прототип** `diploma_nids` объёмом порядка 4 000 строк Python, разделённый на восемь подпакетов с чёткими границами ответственности. Пакет устанавливается через `pip install -e .` без правок в окружении и совместим с Python 3.10+.

2. **Реализованы 15 моделей-кандидатов** в едином регистре через декоратор-фабрику. Все DL-модели возвращают логит `(B,)` для входа `(B, W, F)`; все классические — proba `(B,)` для того же входа после flatten. Это обеспечивает методологически честное сравнение в эксперименте E1.

3. **CNN-LSTM усилен по сравнению с первой версией прототипа**: bidirectional + attention pool + LayerNorm + увеличенный MLP-head. Это даёт реальный шанс достичь NFR-1a (F1 ≥ 0,90) при той же длине окна и тех же гиперпараметрах обучения.

4. **Конечно-автоматный агент работает в двух режимах**: empirical (по построению согласован с маргинальными распределениями UNSW-NB15) и parametric (fallback). Empirical-режим устраняет OOD-проблему первой версии и делает эксперимент E5 методологически корректным.

5. **Подсистема онлайн-инференса полностью готова**: WindowScorer + AlertFormer с пятиступенчатой severity и временной дедупликацией, FastAPI-сервис с 5 эндпоинтами, Streamlit-дэшборд. Все компоненты совместимы по контракту JSON / OpenAPI.

6. **Подсистема мониторинга дрейфа** реализует PSI / KL / MMD с устоявшимися практическими порогами; гауссово ядро MMD использует медианную эвристику автобандвидта (Garreau et al. 2017).

7. **Покрытие тестами — 86 pytest-тестов** на критические инварианты pipeline; все проходят на актуальной среде (Python 3.13.11, PyTorch 2.11 CPU, sklearn 1.8, XGBoost 3.2, Pydantic 2.13).

8. **Реализованы все скрипты экспериментов E1–E6** + cross-evaluation и демонстрационный realtime-прогон. Каждый скрипт — самостоятельный CLI с детальным журналированием и сохранением артефактов в табличном (CSV) и графическом (PNG + PDF) виде.

Результаты Этапа 3 обеспечивают полное основание для перехода к Этапу 4 (экспериментальная верификация, оценка эксплуатационных характеристик, формирование рекомендаций по интеграции).

---

## 15. Источники

<a id="ref-pytorch"></a>**[1]** PyTorch — официальная документация. — [pytorch.org](https://pytorch.org/docs/stable/index.html).

<a id="ref-sklearn"></a>**[2]** Pedregosa F., Varoquaux G., Gramfort A. и др. Scikit-learn: Machine Learning in Python // Journal of Machine Learning Research. 2011. — [scikit-learn.org](https://scikit-learn.org/stable/).

<a id="ref-xgboost"></a>**[3]** Chen T., Guestrin C. XGBoost: A Scalable Tree Boosting System // KDD 2016. — [arXiv:1603.02754](https://arxiv.org/abs/1603.02754); [xgboost.readthedocs.io](https://xgboost.readthedocs.io/).

<a id="ref-pydantic"></a>**[4]** Pydantic v2 — официальная документация. — [docs.pydantic.dev](https://docs.pydantic.dev/latest/).

<a id="ref-fastapi"></a>**[5]** FastAPI — официальная документация. — [fastapi.tiangolo.com](https://fastapi.tiangolo.com/).

<a id="ref-cnnlstm-ieee-2024"></a>**[6]** Enhanced Network Intrusion Detection Using a Hybrid CNN-LSTM Approach on the UNSW-NB15 Dataset // IEEE, 2024. — [IEEE Xplore](https://ieeexplore.ieee.org/document/10770969/).

<a id="ref-scikit-robust"></a>**[7]** scikit-learn — Preprocessing data, RobustScaler. — [scikit-learn.org/stable/modules/preprocessing.html](https://scikit-learn.org/stable/modules/preprocessing.html#scaling-data-with-outliers).

<a id="ref-focal-loss-2017"></a>**[8]** Lin T.-Y., Goyal P., Girshick R., He K., Dollár P. Focal Loss for Dense Object Detection // ICCV 2017. — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002).

<a id="ref-adamw-2017"></a>**[9]** Loshchilov I., Hutter F. Decoupled Weight Decay Regularization // ICLR 2019. — [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).

<a id="ref-temperature-scaling-2017"></a>**[10]** Guo C., Pleiss G., Sun Y., Weinberger K. Q. On Calibration of Modern Neural Networks // ICML 2017. — [arXiv:1706.04599](https://arxiv.org/abs/1706.04599).

<a id="ref-fiddler-psi-2024"></a>**[11]** Measuring Data Drift with the Population Stability Index (PSI) // Fiddler AI Blog. — [fiddler.ai](https://www.fiddler.ai/blog/measuring-data-drift-population-stability-index).

<a id="ref-mmd-median-2014"></a>**[12]** Garreau D., Jitkrittum W., Kanagawa M. Large sample analysis of the median heuristic. — [arXiv:1707.07269](https://arxiv.org/abs/1707.07269).

---

### Связанные документы

- [README.md](README.md) — структура проекта и инструкции запуска.
- [../Stage1/report_stage1.md](../Stage1/report_stage1.md) — обоснование выбора CNN-LSTM и UNSW-NB15.
- [../Stage2/report_stage2.md](../Stage2/report_stage2.md) — детальное проектирование подсистем.
- [../ВКР.txt](../ВКР.txt) — постановка задачи и контекст.
