# RU Liquidity Sentinel

**Индекс ликвидного стресса российского денежного рынка (LSI)**  
Аналитический пайплайн для казначейства ПСБ.

---

## Содержание

- [Описание проекта](#описание-проекта)
- [Архитектура](#архитектура)
- [Структура репозитория](#структура-репозитория)
- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Описание модулей](#описание-модулей)
- [Модели](#модели)
- [Конфигурация](#конфигурация)
- [Дашборд](#дашборд)
- [Тестирование](#тестирование)
- [Артефакты](#артефакты)

---

## Описание проекта

`ru-liquidity-sentinel` — end-to-end ML-пайплайн, рассчитывающий ежедневный **Liquidity Stress Index (LSI)** на основе пяти модулей сигналов российского денежного рынка. Индекс агрегирует данные ЦБ РФ, Минфина, ФНС и Росказны в единый сигнал на шкале `[0, 100]`, где значения выше 70 соответствуют красной зоне (высокий стресс).

**Ключевые особенности:**

- Строго causal-вычисления: нет look-ahead во всех индикаторах
- Rolling-MAD нормализация (окно 365 дней) для робастных z-score
- NSVM (Neural Stochastic Variance Model) — LSTM-архитектура с предсказанием `(mu, log_var)` — основная модель агрегации
- SHAP-декомпозиция по модулям: `contrib_M{n}` суммируется в LSI
- Walk-forward кросс-валидация (5 фолдов, `TimeSeriesSplit`)
- Бэктест на трёх известных эпизодах стресса: декабрь 2014, февраль–апрель 2022, август 2023
- Интерактивный Streamlit-дашборд с тёмной темой

---

## Архитектура

```
Данные ЦБ / Минфин / Росказна / ФНС
        │
        ▼
┌─────────────────────────────────────────┐
│           Слой парсеров (parsers/)       │
│  m1_rreserves  m2_repo  m3_ofz          │
│  m4_tax        m5_bliquidity            │
│                                         │
│  Выход: data/raw/*.csv                  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Инжиниринг признаков           │
│  M1: Резервные требования + RUONIA      │
│  M2: Аукционы РЕПО ЦБ                  │
│  M3: Первичные аукционы ОФЗ (Минфин)   │
│  M4: Налоговая сезонность               │
│  M5: Ликвидность банковского сектора    │
│                                         │
│  Нормализация: rolling MAD z-score      │
│  + бинарные флаги + MIO CUSUM           │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│          Агрегация (aggregate.py)        │
│                                         │
│  NSVMRegressor (LSTM mu + log_var)      │
│  Walk-forward CV + tune_models          │
│  SHAP-атрибуция → share_M{n}           │
│  Log-линейная калибровка → LSI [0,100]  │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│         Оценка и отчётность             │
│  Бэктест 3 эпизодов (backtest.py)       │
│  Hold-out метрики с 2024-01-01          │
│  Auto-detection стресс-эпизодов (top-K) │
│  Noise breakdown по модулям             │
│  EDA + eval_metrics отчёты             │
└──────────────────┬──────────────────────┘
                   │
                   ▼
       Streamlit Dashboard (dashboard/app.py)
```

---

## Структура репозитория

```
lsi-main/
├── src/ru_liquidity_sentinel/   # Основной Python-пакет
│   ├── __init__.py
│   ├── __main__.py              # Точка входа CLI (ru-liquidity-sentinel)
│   ├── config.py                # Все пути, пороги, периоды
│   ├── loaders.py               # Загрузка и нормализация CSV
│   ├── normalize.py             # Rolling MAD z-score, CUSUM, выравнивание
│   ├── pipeline.py              # Оркестратор полного пайплайна
│   ├── aggregate.py             # LSI-агрегация, SHAP-декомпозиция, CV
│   ├── gbm_lsi.py               # Log-калибровка предсказаний → LSI
│   ├── nsvm_model.py            # NSVMRegressor (LSTM, PyTorch)
│   ├── backtest.py              # Оценка на исторических эпизодах
│   ├── reporting.py             # Генерация отчётов (PNG, CSV, MD)
│   └── modules/
│       ├── m1_reserves.py       # M1: Резервные требования + RUONIA-спред
│       ├── m2_repo.py           # M2: Аукционы РЕПО ЦБ
│       ├── m3_ofz.py            # M3: Аукционы ОФЗ (первичный рынок)
│       ├── m4_tax.py            # M4: Налоговая сезонность
│       └── m5_treasury.py       # M5: Ликвидность банксектора / Казначейство
│
├── parsers/                     # Отдельный пакет парсеров (CBR, Minfin, etc.)
│   ├── requirements.txt
│   ├── ru_liquidity_sentinel_parsers/
│   │   ├── config.py
│   │   ├── http.py              # HTTP-клиент с retry (tenacity)
│   │   └── parsers/
│   │       ├── m1_rreserves.py  # Парсер резервных требований ЦБ
│   │       ├── m2_repo.py       # Парсер аукционов РЕПО ЦБ
│   │       ├── m3_ofz.py        # Парсер аукционов ОФЗ Минфин
│   │       ├── m4_tax.py        # Парсер налогового календаря ФНС
│   │       └── m5_treasury.py   # Парсер данных Росказны / ЦБ
│   └── scripts/
│       ├── fetch_all.py         # Загрузка всех источников разом
│       ├── fetch_m1.py … fetch_m5.py
│       └── _cli_helpers.py
│
├── scripts/
│   ├── run_pipeline.py          # Запуск основного пайплайна
│   ├── fetch_and_run.py         # Fetch + pipeline за один шаг
│   ├── eda.py                   # Exploratory Data Analysis
│   └── eval_metrics.py          # Расчёт и сохранение метрик
│
├── dashboard/
│   └── app.py                   # Streamlit-дашборд
│
├── tests/
│   └── test_pipeline.py
│
├── data/raw/                    # CSV-файлы (gitignored, генерируются парсерами)
├── artifacts/                   # Parquet / CSV с результатами пайплайна
├── reports/
│   ├── EDA.md
│   ├── figures/                 # PNG-графики
│   └── tables/                  # CSV-таблицы метрик
│
├── pyproject.toml               # Сборка пакета и зависимости
├── Makefile                     # Удобные команды make
└── requirements.txt             # Единый файл зависимостей (этот файл)
```

---

## Установка

### Требования

- Python **≥ 3.10**
- (опционально) CUDA-совместимый GPU для ускорения NSVM

### Вариант 1 — через `requirements.txt` (рекомендуется)

```bash
git clone <repo-url> school_AI
cd school_AI
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### Вариант 2 — через `pyproject.toml` (все компоненты)

```bash
pip install -e ".[models,dashboard,dev]"
cd parsers && pip install -r requirements.txt
```

### Вариант 3 — через Makefile

```bash
make install          # установка парсеров + пайплайна
```

---

## Быстрый старт

```bash
# 1. Загрузка данных из открытых источников (CBR, Minfin, ФНС, Росказна)
make fetch
# или
python scripts/fetch_and_run.py   # fetch + pipeline за один вызов

# 2. Запуск полного пайплайна (обучение модели + расчёт LSI)
make pipeline
# или
python scripts/run_pipeline.py

# 3. Генерация аналитических отчётов (EDA + метрики)
make reports

# 4. Запуск дашборда
make dashboard
# → открыть http://localhost:8501

# Шаги 1–3 одной командой
make all
```

### CLI

После установки доступен CLI-интерфейс:

```bash
ru-liquidity-sentinel --help
```

---

## Описание модулей

| Модуль | Источник | Ключевые сигналы | Особенности |
|--------|----------|-----------------|-------------|
| **M1** — Резервные требования | ЦБ РФ | Спред фактических / обязательных резервов, RUONIA − ключевая ставка | Флаг конца периода усреднения |
| **M2** — Аукционы РЕПО | ЦБ РФ | Объём, ставка, спред к ключевой ставке (7-дневный срок) | Флаги недельного цикла РЕПО |
| **M3** — Аукционы ОФЗ | Минфин | Объём размещения, bid-to-cover, доходность | Флаги аукционных дней |
| **M4** — Налоговая сезонность | ФНС / Минфин | Количество налоговых событий, пики выплат | Мультипликатор `[1.0, 1.4]`, флаги EoM / EoQ / EoY |
| **M5** — Ликвидность банксектора | ЦБ РФ, Росказна | Дефицит ликвидности, корсчета, депозиты Казначейства | Дельты Δ1d / Δ5d / Δ22d |

Все модули применяют единую нормализацию:

```
mad_score = (x − медиана) / (1.4826 × MAD)   [365-дневное окно]
```

плюс бинарные флаги (сезонные события) и **MIO CUSUM** — онлайн-детектор аномалий без смотра в будущее.

---

## Модели

### NSVMRegressor (единственная модель агрегации, `nsvm_model.py`)

Нейросетевая двухголовочная архитектура на базе LSTM (PyTorch) — **единственный** агрегатор LSI:

```
Вход: окно seq_len=14 дней × N признаков
     ↓
LSTM (mean path, hidden=64)    LSTM (vol path, hidden=32)
     ↓                               ↓
Linear(64→32)→ReLU→Linear(32→1)  Linear(32→16)→ReLU→Linear(16→1)
     ↓                               ↓
    mu (предсказание стресса)    log_var (неопределённость)
```

- Функция потерь: **NLL (отрицательное логарифмическое правдоподобие)** для гетероскедастичных данных
- Обёртка `sklearn.BaseEstimator` — совместима с `TimeSeriesSplit`
- SHAP-атрибуция через `GradientExplainer` (суммирование по временной оси `seq_len`)
- Поддерживает **`partial_fit`**: дообучение только head-слоёв (LSTM заморожен) для адаптации к структурным сдвигам

### Калибровка LSI

Предсказания NSVM (`mu`) переводятся в `[0, 100]` через лог-линейную калибровку по двум якорям (q10 / q99 train-сета):

```
LSI = clip(a × log(pred − q_low + ε) + b, 0, 100)
```

### Пороги статуса

| Зона | LSI |
|------|-----|
| 🟢 Зелёная (норма) | < 40 |
| 🟡 Жёлтая (внимание) | 40–70 |
| 🔴 Красная (стресс) | > 70 |

---

## Конфигурация

Все параметры сосредоточены в `src/ru_liquidity_sentinel/config.py` и переопределяются через `PipelineConfig`:

```python
from ru_liquidity_sentinel.config import PipelineConfig
from ru_liquidity_sentinel.pipeline import run_pipeline

cfg = PipelineConfig(
    mad_window_days=365,          # Окно MAD (дни)
    model_test_split_date="2024-01-01",
    cv_n_splits=5,
    tune_models=True,
    early_stopping_rounds=50,
    n_estimators_max=1500,
    episode_min_days=5,
    episode_merge_gap_days=5,
    episode_top_k=25,
)
run = run_pipeline(cfg)
```

---

## Дашборд

Streamlit-приложение (`dashboard/app.py`) предоставляет:

- **Временной ряд LSI** с цветовой зонировкой (зелёная / жёлтая / красная)
- **Декомпозиция по модулям** — `contrib_M1..M5` с объяснением вклада каждого
- **История стресс-эпизодов** и бэктест на трёх ключевых кризисах
- **Breakdown шума** — разложение Var(ΔLSI) на компоненты модулей
- **Feature importance** и SHAP-plots

```bash
streamlit run dashboard/app.py
```

---

## Тестирование

```bash
make test
# или
pytest -q tests/
```

Тесты покрывают: загрузку данных, построение признаков, выходные форматы модулей M1–M5, корректность индекса дат.

---

## Артефакты

После успешного `run_pipeline()` в `artifacts/` появляются:

| Файл | Описание |
|------|----------|
| `features_daily.parquet` | Дневная матрица признаков (все модули) |
| `lsi_daily.parquet` | LSI + `contrib_M{n}` + `share_M{n}` |
| `proxy_target.parquet` | Proxy-целевая переменная (без look-ahead) |
| `backtest_episodes.csv` | Метрики на 3 исторических эпизодах |
| `auto_episodes.csv` | Автоматически найденные эпизоды стресса (top-25) |
| `noise_breakdown.csv` | Var(ΔLSI) по модулям |
| `model_predictions.parquet` | Предсказания NSVM (mu — детерминированная компонента) |
| `model_metrics.csv` | Train / test метрики (MAE, R², и др.) |
| `model_best_params.csv` | Лучшие гиперпараметры по результатам walk-forward CV |
| `model_nsvm_module_attribution.parquet` | SHAP-атрибуция по модулям для NSVM |
| `sensitivity.parquet` | Детальные результаты анализа чувствительности (±20% гиперпараметров) |
| `sensitivity_summary.csv` | Сводная таблица анализа чувствительности |
| `metadata.json` | Мета-информация прогона (даты, квантили LSI, параметры) |

---

## Источники данных

| Источник | Данные | Модуль |
|----------|--------|--------|
| ЦБ РФ (cbr.ru) | RUONIA, ключевая ставка, РЕПО, резервы, ликвидность банксектора | M1, M2, M5 |
| Минфин РФ (minfin.gov.ru) | Аукционы ОФЗ | M3 |
| ФНС РФ (nalog.gov.ru) | Налоговый календарь | M4 |
| Росказна (roskazna.gov.ru) | Депозиты ЕКС, индекс документов | M5 |

---

## Автор

**ImPaul** — ML-инженер, казначейство ПСБ.
