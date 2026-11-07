#!/usr/bin/env python3
"""Whole pipeline on real CDEK points in one command (called by ./run.sh).

1. pick the best available points file (or --points);
2. load it (CDEK API JSON / Overpass JSON / CSV), keep points inside the walking graph;
3. save the normalised table to data/results/points_used.csv;
4. scenarios 1 and 2 for the author's real commutes → data/results/scenario*_real.json;
5. experiment on synthetic users (scripts/05_experiment.py);
6. Leaflet demo (scripts/09_export_demo.py) → demo/index.html.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import load_graph  # noqa: E402
from parcel_on_the_way.points import WEEKDAYS, load_any_points, snap_landmarks  # noqa: E402
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402
from parcel_on_the_way.scenarios import (  # noqa: E402
    DetourEngine,
    scenario1_pick_point,
    scenario2_pickup_on_way,
)
from parcel_on_the_way.sources import filter_in_graph  # noqa: E402

RESULTS = ROOT / "data" / "results"

# First existing file wins: exact CDEK data first, OSM draft last.
# cdek.ru site export (510 points, 2026-10-16) beats the API JSON: the API file
# comes from the edu sandbox and holds a stale 142-point subset.
POINT_SOURCES = [
    ROOT / "data" / "cdek_real" / "cdek_moscow_points.csv",
    ROOT / "data" / "cdek_real" / "cdek_api_moscow_raw.json",
    ROOT / "data" / "cdek_real" / "osm_moscow_raw.json",
    ROOT / "data" / "cdek_points_draft_in_area.csv",
]


def pick_source(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit)
        return path if path.is_absolute() else ROOT / path
    for path in POINT_SOURCES:
        if path.exists() and path.stat().st_size > 0:
            return path
    raise SystemExit("Нет файла с пунктами: положите выгрузку в data/cdek_real/ или укажите --points")


def write_points_csv(points, path: Path) -> None:
    """Normalised table in the CDEK template format (read back by every script)."""
    cols = ["id", "address", "type"] + [f"hours_{d}" for d in WEEKDAYS] + [
        "max_weight_kg", "source_url", "checked_date", "lat", "lon", "snap_dist_m", "note",
    ]
    today = date.today().isoformat()
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for p in points:
            hours = []
            for d in WEEKDAYS:
                start, end = p.hours_on(d)
                hours.append("closed" if start is None else f"{start}-{end}")
            w.writerow(
                [p.id, p.name, "постамат" if p.point_type == "postamat" else "ПВЗ", *hours,
                 f"{p.max_weight_kg:g}", p.note, today, f"{p.lat:.7f}", f"{p.lon:.7f}",
                 f"{p.snap_dist_m:.1f}", p.note]
            )


def run_step(title: str, args: list[str]) -> None:
    print(f"\n### {title}", flush=True)
    t0 = time.perf_counter()
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)
    print(f"### {title}: {time.perf_counter() - t0:.0f} с", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", help="файл с пунктами (JSON API СДЭК, Overpass JSON или CSV)")
    ap.add_argument("--date", default="2026-10-14", help="день сценариев, YYYY-MM-DD")
    ap.add_argument("--weight", type=float, default=2.0, help="вес посылки, кг")
    ap.add_argument("--users", type=int, default=200, help="синтетических пользователей")
    ap.add_argument("--max-snap-m", type=float, default=250.0,
                    help="дальше этого от пешеходного графа пункт отбрасывается")
    ap.add_argument("--skip-experiment", action="store_true")
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    src = pick_source(args.points)
    print(f"Пункты: {src.relative_to(ROOT) if src.is_relative_to(ROOT) else src}")
    if src.name.startswith("cdek_points_draft"):
        print("  ВНИМАНИЕ: это черновик из OSM, часы работы не проверены (unknown → 10:00–20:00).")

    t0 = time.perf_counter()
    graph = load_graph()
    print(f"Граф: {graph.number_of_nodes()} узлов, {graph.number_of_edges()} рёбер "
          f"({time.perf_counter() - t0:.0f} с)")

    all_points = load_any_points(src)
    points, dropped = filter_in_graph(graph, all_points, args.max_snap_m)
    unknown = sum("unknown" in p.note for p in points)
    print(f"Загружено {len(all_points)}, в зоне графа {len(points)} "
          f"(ПВЗ {sum(p.point_type == 'pvz' for p in points)}, "
          f"постаматов {sum(p.point_type == 'postamat' for p in points)}), "
          f"вне зоны {len(dropped)}, без точных часов {unknown}")
    if not points:
        raise SystemExit("В зоне графа нет ни одного пункта")
    used_csv = RESULTS / "points_used.csv"
    write_points_csv(points, used_csv)
    print(f"Сохранено: {used_csv.relative_to(ROOT)}")

    cfg = load_habitual_config()
    landmarks = {k: info["node"] for k, info in snap_landmarks(graph).items()}
    engine = DetourEngine(graph, landmarks, cfg)

    s1 = scenario1_pick_point(engine, points, day=args.date, parcel_weight_kg=args.weight)
    (RESULTS / "scenario1_real.json").write_text(
        json.dumps(s1, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСценарий 1 ({args.date}, {args.weight:g} кг): кандидатов {s1['n_candidates']}")
    for r in s1["results"]:
        print(f"  • {r['point_name']}: {r['reason_ru']}")

    if s1["results"]:
        top = s1["results"][0]["point_id"]
        s2 = scenario2_pickup_on_way(engine, points, top, day=args.date, parcel_weight_kg=args.weight)
        (RESULTS / "scenario2_real.json").write_text(
            json.dumps(s2, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Сценарий 2 ({s2.get('point_name')}): {s2.get('notification_ru')}")

    if not args.skip_experiment:
        run_step("Эксперимент", ["scripts/05_experiment.py", "--csv", str(used_csv),
                                 "--n", str(args.users), "--parcel-kg", str(args.weight)])
    run_step("Демо-карта", ["scripts/09_export_demo.py", "--csv", str(used_csv)])

    print("\nГотово. Результаты: data/results/, карта: demo/index.html")


if __name__ == "__main__":
    main()
