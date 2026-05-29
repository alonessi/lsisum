# RU-Liquidity-Sentinel — парсеры данных по 5 модулям

Скрипты Python 3.11 для сбора данных, описанных в ТЗ «Раннее предупреждение
дефицита/профицита ликвидности на денежном рынке РФ». Это **только сбор
сырых данных и нормализация в CSV** — без ML / LSI / дашбордов / LLM.

## Установка

```bash
cd /home/ubuntu/ru-liquidity-sentinel
python3 -m pip install -r requirements.txt
```

Поддерживается Python 3.11+.

## Источники и модули

| Модуль | Источник | Что собираем | CSV-выход |
|--------|----------|--------------|-----------|
| **M1** | ЦБ — `RReserves/required_reserves_table.xlsx`, `hd_base/ruonia/dynamics/` | Обязательные резервы (фактические/требуемые остатки) + RUONIA daily | `m1_rreserves.csv`, `m1_ruonia.csv` |
| **M2** | ЦБ — `hd_base/repo/`, `hd_base/keyrate/` | Аукционы РЕПО + ключевая ставка | `m2_repo_auctions.csv`, `m2_keyrate.csv` |
| **M3** | Минфин — `perfomance/public_debt/internal/operations/ofz/auction` (yearly XLSX) | ОФЗ-аукционы 2015–2026 (предложение/спрос/размещение, доходности, cover ratio, флаги пере-/недо-спроса) | `m3_ofz_auctions.csv` |
| **M4** | ФНС — open data набор «Налоговый календарь» `nalog.gov.ru/opendata/7707329152-kalendar/` (XML-снапшоты с 2014 по 2026) | Налоговый календарь и дневные флаги (пики 13-16/18-22/23-25/25-28, конец месяца/квартала/года) | `m4_tax_calendar.csv`, `m4_tax_flags_daily.csv` |
| **M5** | ЦБ — `hd_base/bliquidity/`, СОРС `vfs/statistics/BankSector/Borrowings/02_01_Funds_all.xlsx`; Росказна — индекс операционных дней по депозитам ЕКС | Дефицит/профицит ликвидности банковского сектора (с 2014 г.), привлечённые средства организаций (всего/инвалюта/итого), индекс документов Росказны по размещению ЕКС на банковских депозитах | `m5_bliquidity.csv`, `m5_sors_funds_all.csv`, `m5_roskazna_eks_deposits_index.csv` |

## Запуск

Все скрипты — обычные CLI на `click`. По умолчанию глубина истории
`--from 2014-01-01` до сегодня.

```bash
# Все модули за всю историю
python3 scripts/fetch_all.py

# Любой отдельный модуль
python3 scripts/fetch_m1.py --from 2014-01-01 --to 2026-05-09
python3 scripts/fetch_m2.py --detailed                  # +per-аукцион детали
python3 scripts/fetch_m3.py
python3 scripts/fetch_m4.py
python3 scripts/fetch_m5.py --roskazna-pages 50

# Если ФНС публикует ещё один XML-снапшот, не попавший в discovery,
# его можно подмешать вручную:
python3 scripts/fetch_m4.py \
  --extra-xml-url https://data.nalog.ru/opendata/7707329152-kalendar/data-XYZ.xml

# Пропустить выбранные модули в общем запуске
python3 scripts/fetch_all.py --skip-m4 --skip-m5
```

Опции у `fetch_all.py`:

```
--from / --to              диапазон дат
--detailed-repo            тяжёлый pull детальных карточек репо-аукционов (медленно)
--skip-m1 .. --skip-m5     пропустить модуль
--m4-extra-xml URL          доп. XML-снапшот ФНС (можно несколько раз)
--roskazna-pages N         сколько страниц индекса Росказны крутить (по 10 дней)
--out-raw, --out-processed переопределить директории (по умолчанию data/raw, data/processed)
```

## Структура проекта

```
ru-liquidity-sentinel/
├── README.md
├── requirements.txt
├── ru_liquidity_sentinel/
│   ├── config.py           # пути, дефолтные даты, HTTP-настройки
│   ├── http.py             # requests.Session + retry/backoff (tenacity)
│   └── parsers/
│       ├── _common.py      # parse_ru_number/parse_ru_date/write_csv
│       ├── m1_rreserves.py
│       ├── m2_repo.py
│       ├── m3_ofz.py
│       ├── m4_tax.py
│       └── m5_treasury.py
├── scripts/
│   ├── _cli_helpers.py
│   ├── fetch_all.py
│   ├── fetch_m1.py
│   ├── fetch_m2.py
│   ├── fetch_m3.py
│   ├── fetch_m4.py
│   └── fetch_m5.py
└── data/
    ├── raw/                # сырые HTML/XLSX по источникам, для повторной обработки
    └── processed/          # нормализованные CSV (UTF-8 BOM, ISO-даты)
```

## Формат CSV

- Кодировка: UTF-8 with BOM (Excel-friendly).
- Разделитель: `,`.
- Даты: `YYYY-MM-DD`.
- Числа: точка как десятичный разделитель, без разделителей тысяч.
- При повторном запуске CSV полностью перезаписывается; сырые HTML/XLSX
  складываются по источнику в `data/raw/<источник>/...`.

## Замечания по покрытию данных

- **M1 RReserves**: с 2014 г. (148 позиций, по периодам усреднения).
- **M1 RUONIA**: с 2014 г. ежедневно.
- **M2 REPO/keyrate**: с 2014 г. ежедневно (398 уникальных аукционов на период).
- **M3 OFZ**: 2015–2026; за 2014 на сайте Минфина агрегированный XLSX
  отсутствует (раньше публиковались только отдельные пресс-релизы).
- **M4 ФНС**: парсятся 16 XML-снапшотов из официального open-data набора
  ФНС (`nalog.gov.ru/opendata/7707329152-kalendar/`). Покрытие — 2014, 2015,
  2016, 2018-2026 (≈1070 уникальных событий). 2017 пропущен в самом
  open-data наборе (между снапшотами `data-20160324...` за 2016 г. и
  `data-20171219...` за 2018 г. ФНС не выкладывал XML для 2017 г.).
  Daily-флаги (`flag_tax_peak_*`, `flag_end_of_month`, `flag_quarter_end`,
  `flag_year_end`) считаются программно за весь запрошенный диапазон —
  они не зависят от наличия XML.

  Парсер устойчив к двум типичным проблемам этого источника: (а) у
  снапшота за 2023 г. дублируется хвост документа после `</calendar>`,
  (б) у снапшота за 2026 г. файл фактически в UTF-8 при объявленной
  `windows-1251` кодировке. Оба случая обрабатываются автоматически.
- **M5 bliquidity**: с 2014 г. (3 090+ дневных строк, 15 показателей).
- **M5 СОРС funds_all**: содержит ряды с 2019 г. (XLSX обновляется ЦБ еженедельно).
- **M5 Росказна**: индекс операционных дней по депозитам ЕКС;
  агрегированной таблицы с суммами нет — индивидуальные суммы и сроки
  лежат внутри `.docx`/`.XML` файлов (не парсятся в этой версии,
  но URL-ы сохраняются для последующего извлечения).
  SSL: сайт Росказны использует сертификат «Минцифры», поэтому
  `verify=False` (см. `m5_treasury.fetch_roskazna_index`).

## Идемпотентность и устойчивость

- `requests.Session` с реалистичным User-Agent.
- Ретраи 5 раз с экспоненциальной задержкой через `tenacity`
  (только сетевые ошибки и 5xx — см. `ru_liquidity_sentinel/http.py`).
- Сырые ответы сохраняются в `data/raw/`, чтобы CSV можно было
  пересобрать без повторного похода на источник.

## Quick-check примерных объёмов

После `python3 scripts/fetch_all.py`:

| CSV | Строки | Колонки |
|-----|--------|---------|
| `m1_rreserves.csv` | 148 | 11 |
| `m1_ruonia.csv` | 3 028 | 11 |
| `m2_repo_auctions.csv` | 399 | 7 |
| `m2_keyrate.csv` | 3 092 | 2 |
| `m3_ofz_auctions.csv` | 1 022 | 22 |
| `m4_tax_calendar.csv` | 1 071 | 4 |
| `m4_tax_flags_daily.csv` | 4 748 | 14 |
| `m5_bliquidity.csv` | 3 095 | 20 |
| `m5_sors_funds_all.csv` | 3 051 | 4 |
| `m5_roskazna_eks_deposits_index.csv` | 100 | 5 |

(объёмы плавают на единицы строк по мере выхода новой данных).
