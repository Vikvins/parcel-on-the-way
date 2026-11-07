#!/usr/bin/env python3
"""Steps 4–5: scenario API demo + service-time example + unit-test run."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import load_graph  # noqa: E402
from parcel_on_the_way.hours import is_open_at  # noqa: E402
from parcel_on_the_way.points import (  # noqa: E402
    load_pickup_points,
    snap_landmarks,
    snap_points,
)
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402
from parcel_on_the_way.scenarios import (  # noqa: E402
    DetourEngine,
    scenario1_pick_point,
    scenario2_pickup_on_way,
)

OUT_DIR = ROOT / "data" / "processed"


def service_time_example(points):
    print("\n===== Service-time filter example =====")
    p = next(x for x in points if x.id == "PVZ-TEST-07")  # closes 18:00
    print(f"Point {p.id}: fri/wed hours {p.hours_on('wed')} (closes 18:00)")
    for arrive, service in [("17:56", 4.0), ("17:58", 4.0), ("17:58", 2.0), ("18:00", 4.0)]:
        ok, reason = is_open_at(p, "wed", arrive, service)
        leave = f"leave≈{arrive[:3]}{int(arrive[3:]) + int(service):02d}"
        print(f"  arrive {arrive}, service {service:.0f} min → {'OPEN' if ok else 'CUT '}  ({reason})")
    print("  → cut ONLY because of service time: arrive 17:58 with service 4 min")
    print("    (arrival 17:58 is before 18:00, but 17:58+4=18:02 > close 18:00)")


def main():
    cfg = load_habitual_config()
    g = load_graph()
    points = load_pickup_points()
    snap_points(g, points)
    landmarks = {k: info["node"] for k, info in snap_landmarks(g).items()}
    engine = DetourEngine(g, landmarks, cfg)

    print(f"Graph {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")
    print(f"service_time_min={cfg.service_time_min}")
    print(f"routes: {[r.id for r in cfg.routes]}")

    service_time_example(points)

    print("\n===== Scenario 1 (Wednesday 2026-10-14) =====")
    s1 = scenario1_pick_point(engine, points, day="2026-10-14", parcel_weight_kg=2.0)
    print(json.dumps(s1, ensure_ascii=False, indent=2))
    (OUT_DIR / "scenario1_wed.json").write_text(
        json.dumps(s1, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== Scenario 2: PVZ-TEST-07 on Wednesday =====")
    s2 = scenario2_pickup_on_way(
        engine, points, point_id="PVZ-TEST-07",
        day="2026-10-14", parcel_weight_kg=2.0,
    )
    print(json.dumps(s2, ensure_ascii=False, indent=2))
    (OUT_DIR / "scenario2_pvz07_wed.json").write_text(
        json.dumps(s2, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n===== Scenario 2: PVZ-TEST-07 on a day it never works (weight cut) =====")
    s2b = scenario2_pickup_on_way(
        engine, points, point_id="PVZ-TEST-10",
        day="2026-10-14", parcel_weight_kg=8.0,
    )
    print(json.dumps(s2b, ensure_ascii=False, indent=2))

    print(f"\nSaved JSON under {OUT_DIR}/")


if __name__ == "__main__":
    main()
