# RU Liquidity Sentinel

RU Liquidity Sentinel считает Liquidity Stress Index (LSI) для российского денежного рынка. Финальная версия использует NSVM-агрегатор, ECDF-калибровку и инкрементальное обновление артефактов для Streamlit.

README описывает актуальный контур проекта. Экспериментальные отчёты и прежние агрегаторы удалены из финальной версии.

## Архитектура

Рабочий процесс разделён на две фазы.

1. Первичная инициализация выполняется оффлайн:
   - загружаются CSV из `data/raw`;
   - строятся признаки M1-M5;
   - NSVM обучается на истории до `PipelineConfig.nsvm_init_cutoff_date`;
   - ECDF-калибратор обучается на сглаженном NLL NSVM;
   - LSI рассчитывается для всей доступной истории;
   - сохраняются `historical_lsi.parquet`, `nsvm_weights.pt`, `calibrator.pkl`.

2. Инкрементальное обновление выполняется из CLI или кнопкой в Streamlit:
   - свежие данные подтягиваются парсерами;
   - загружаются веса NSVM и ECDF-калибратор;
   - новые дни прогнозируются через `.predict()` с историческим контекстом `seq_len`;
   - новые строки LSI дописываются в `historical_lsi.parquet`;
   - если средний NLL новых дней выше порога режима, запускается `.partial_fit()` только на новых данных с хвостом прошлой истории;
   - обновлённые веса перезаписывают `nsvm_weights.pt`.

## Экономическая логика модулей

`M1` оценивает резервное усреднение и RUONIA. Стрессом считается дефицит резервов: `required_avg_blnrub - actual_balances_blnrub`, а не положительный запас ликвидности.

`M2` оценивает аукционы РЕПО Банка России. Спрос из `bliq_repo_auction_vol` переводится из млрд в млн рублей перед расчётом cover ratio, чтобы масштаб совпадал с `allotment_mlnrub`.

`M3` оценивает первичные аукционы ОФЗ. Недоподписка и рост доходности являются стрессом; высокая переподписка сама по себе не считается стрессом.

`M4` описывает налоговую сезонность. Флаги налоговых периодов не подаются как обычный стресс-вход NSVM, а используются для сезонного контекста и штрафа предсказуемого налогового давления.

`M5` оценивает казначейские потоки и банковскую ликвидность. Для MIO используется `-delta_5d`: отрицательный пятидневный поток трактуется как бюджетный drain.

## Установка

```powershell
py -m pip install -r requirements.txt
```

Если `py` на машине не настроен, используйте путь к нужному Python-интерпретатору, но команды ниже предполагают Windows launcher `py`.

## Данные

Основной код читает CSV из `data/raw`.

Чтобы обновить данные через парсеры:

```powershell
cd parsers
py scripts/fetch_all.py
cd ..
copy parsers\data\processed\*.csv data\raw\
```

То же самое через Makefile:

```powershell
make fetch
```

## Запуск

Первичная инициализация:

```powershell
py scripts/incremental_nsvm.py --init
```

Инициализация с исследовательским cutoff:

```powershell
py scripts/incremental_nsvm.py --init --cutoff 2024-01-01
```

Инкрементальное обновление с загрузкой свежих данных:

```powershell
py scripts/incremental_nsvm.py
```

Инкрементальное обновление без запуска парсеров:

```powershell
py scripts/incremental_nsvm.py --no-fetch
```

Streamlit:

```powershell
py -m streamlit run dashboard/app.py
```

Тесты:

```powershell
py -m pytest -q
```

## Артефакты

Финальные артефакты лежат в `artifacts/`.

Ключевые файлы:

- `historical_lsi.parquet` - основная история LSI;
- `nsvm_weights.pt` - веса NSVM и список признаков;
- `calibrator.pkl` - ECDF-калибратор;
- `incremental_metadata.json` - служебное состояние incremental-контура;
- `features_daily.parquet` - дневная матрица признаков M1-M5;
- `model_predictions.parquet` - актуальная линия NSVM для dashboard;
- `model_nsvm_module_attribution.parquet` - вклад M1-M5;
- `sensitivity.parquet` - траектории LSI при perturbation гиперпараметров;
- `sensitivity_summary.csv` - сводка чувствительности по mean/max/q90 отклонениям и корреляции;
- `backtest_episodes.csv`, `auto_episodes.csv`, `noise_breakdown.csv`, `model_metrics.csv` - диагностические таблицы для Streamlit.

Если код модели менялся после генерации артефактов, Streamlit покажет предупреждение. В этом случае пересоберите историю:

```powershell
py scripts/incremental_nsvm.py --init
```

## Структура проекта

```text
dashboard/
  app.py                         Streamlit-интерфейс и кнопка incremental update
scripts/
  incremental_nsvm.py            CLI для init/update
src/ru_liquidity_sentinel/
  aggregate.py                   сборка LSI, вклады, эпизоды, чувствительность
  backtest.py                    проверка исторических стресс-эпизодов
  config.py                      пути, пороги, даты и параметры NSVM
  features.py                    сборка дневной feature frame M1-M5
  incremental.py                 offline init и incremental update
  loaders.py                     загрузка CSV
  normalize.py                   MAD, winsorization, CUSUM
  nsvm_lsi.py                    ECDF-калибровка и атрибуция NSVM
  nsvm_model.py                  NSVM anomaly detector
  modules/                       экономические модули M1-M5
tests/                           smoke и экономические direction-тесты
parsers/                         загрузчики внешних источников
```

## Makefile

На Windows можно использовать короткие команды:

```powershell
make install
make fetch
make init
make update
make dashboard
make test
```
