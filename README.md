# Посылка по пути

Прототип модуля для приложения СДЭК: подбор пункта выдачи с **наименьшим крюком**
от привычного пешеходного маршрута.

**Демо-карта:** https://vikvins.github.io/parcel-on-the-way/ (исходник в `docs/`, собирается `scripts/09_export_demo_redesign.py`).

Работа подана на стипендиальный конкурс «Система» 2026/27, номинация «Технологии в логистике».

## Быстрый старт (одна команда)

```bash
git clone https://github.com/Vikvins/parcel-on-the-way
cd parcel-on-the-way
./run.sh --open
```

Скрипт сам создаёт `.venv`, ставит зависимости (один раз), при отсутствии строит
пешеходный граф, затем:

1. берёт лучший доступный файл пунктов СДЭК (первый найденный):
   `data/cdek_real/cdek_moscow_points.csv` (сайт cdek.ru, 510 пунктов Москвы, 16.10.2026) →
   `data/cdek_real/cdek_api_moscow_raw.json` (API СДЭК, тестовый контур — устаревшая выборка) →
   `data/cdek_real/osm_moscow_raw.json` (OSM) →
   `data/cdek_points_draft_in_area.csv` (черновик OSM, часы не проверены);
2. отбрасывает пункты вне зоны пешеходного графа (дальше 250 м от сети);
3. считает сценарии 1 и 2 для реальных маршрутов автора, эксперимент на
   200 синтетических пользователях и собирает карту `demo/index.html`.

| Команда | Что делает |
| --- | --- |
| `./run.sh` | весь конвейер, ~3 мин |
| `./run.sh --open` | то же и открыть карту в браузере |
| `./run.sh --points путь/к/файлу.json` | свой файл пунктов (JSON API СДЭК, Overpass JSON или CSV) |
| `./run.sh --date 2026-10-16 --weight 5` | другой день и вес посылки |
| `./run.sh --users 50 --skip-experiment` | быстрый прогон без эксперимента |
| `./run.sh test` | тесты |

Результаты в `data/results/`: `points_used.csv` (пункты, на которых считали),
`scenario1_real.json`, `scenario2_real.json`, `experiment_summary.csv`,
`experiment_user_results.csv`, графики `*.png`.

Форматы пунктов, которые понимает загрузчик (`src/parcel_on_the_way/sources.py`):

- **API СДЭК v2** `/v2/deliverypoints`: `work_time_list` или `work_time`
  («Пн-Пт 10:00-21:00, Сб-Вс 10:00-18:00»), `weight_max`, `type` PVZ/POSTAMAT;
  пункты с `is_handout: false` (только приём) пропускаются.
- **Overpass JSON** с тегом `opening_hours` (`Mo-Fr 10:00-21:00; Sa,Su 10:00-18:00`, `24/7`).
- **CSV** с колонками `id, address, type, hours_mon … hours_sun, lat, lon`
  (+ необязательные `max_weight_kg`, `source_url`); часы `10:00-21:00`,
  пусто/`closed` = выходной, `00:00-24:00`/`круглосуточно`, `unknown` = 10:00–20:00 с пометкой.

Метрика задержки в эксперименте: `mean_days_to_pickup` учитывает и отдельную
поездку, поэтому почти всегда 0; `mean_days_on_way` и `share_on_way_le_1` считают
только забор **по пути** (крюк ≤ 5 мин) — это честная цена «посылки по пути».

---

Формула крюка:

```text
detour(P) = t(A→P) + t(P→B) − t(A→B)
```

## Данные

- **Площадка**: прямоугольник ~5.2 × 2.2 км вдоль Ленинского проспекта
  (МИСИС, Ленинский 4 / окрестности м. «Октябрьская» ↔ дом у пр-та 60-летия Октября /
  окрестности м. «Академическая»).
- **Граф**: OpenStreetMap, `network_type=walk`, OSMnx, CRS EPSG:32637.
- **Скорость ходьбы**: 5 км/ч (83.33 м/мин) для всех рёбер.
- `data/pickup_points.csv` — **вымышленные** пункты для тестов; реальные — в `data/cdek_real/`
  и `data/cdek_points_draft_in_area.csv`.

## Пошаговый запуск (для отладки)

```bash
cd parcel-on-the-way
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt   # pytest

# 1. Граф коридора
PYTHONPATH=src .venv/bin/python scripts/01_build_graph.py
PYTHONPATH=src .venv/bin/python scripts/01b_plot_corridor.py

# 2. Остановки + пункты + снаппинг
PYTHONPATH=src .venv/bin/python scripts/02_make_sample_points.py

# 3. Ядро крюка (два Dijkstra)
PYTHONPATH=src .venv/bin/python scripts/03_detour_core.py

# 4–5. API сценариев
PYTHONPATH=src .venv/bin/python scripts/04_scenarios_api.py

# 7. Эксперимент (200 синтетических пользователей)
PYTHONPATH=src .venv/bin/python scripts/05_experiment.py \
    --csv data/pickup_points.csv --n 200 --seed 20261015

# Тесты
PYTHONPATH=src .venv/bin/python -m pytest tests/ -v
```

Для **реальных** пунктов СДЭК: положите CSV с теми же колонками и укажите `--csv path/to/points.csv`.
Часы работы — по дням недели; пустые поля = закрыт. Значения departure_time в
`data/habitual_routes.yaml` — примеры; диапазоны для эксперимента там же.

Артефакты:

| Путь | Что |
| --- | --- |
| `data/processed/moscow_walk.graphml` | Граф с `length` и `travel_time_min` |
| `data/processed/moscow_walk.gpkg` | Слои `nodes`, `edges` |
| `data/processed/corridor_map.png` | Диагностическая карта |
| `data/raw/metadata.json` | Параметры сборки |

## Структура

```text
src/parcel_on_the_way/   # библиотека (пакет)
scripts/                 # пошаговые запуски
data/                    # OSM-кеш, граф, CSV пунктов, результаты
demo/                    # карта-демо Leaflet
tests/                   # pytest
run.sh                   # запуск всего одной командой
```

## Язык

Код и имена модулей — английский. README и подписи в демо — русский.
