#!/usr/bin/env python3
"""Step 3: detour core (two Dijkstra) + arrival time + open/weight filters.

Demo tables for one morning and one evening commute (default weekday: wed).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.detour import (  # noqa: E402
    arrival_offset_min,
    best_leg_for_point,
    commute_timeline,
    detour_for_point,
    score_leg,
)
from parcel_on_the_way.graph import load_graph, snap_to_node  # noqa: E402
from parcel_on_the_way.hours import is_open_at, minutes_to_hhmm, weight_ok  # noqa: E402
from parcel_on_the_way.points import (  # noqa: E402
    load_pickup_points,
    snap_landmarks,
    snap_points,
)
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402

OUT_CSV = ROOT / "data" / "processed" / "detour_demo_wed.csv"


def format_table(rows: list[dict], title: str) -> None:
    print(f"\n===== {title} =====")
    header = (
        f"{'point':18s} {'leg':28s} {'tA→P':>6s} {'tP→B':>6s} {'detour':>7s} "
        f"{'arrive':>7s} {'open':>4s} {'wt':>4s} {'on_path':>7s}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['point_id']:18s} {r['leg']:28s} "
            f"{r['t_a_to_p']:6.2f} {r['t_p_to_b']:6.2f} {r['detour']:7.2f} "
            f"{r['arrival']:>7s} {'YES' if r['open'] else 'no':>4s} "
            f"{'YES' if r['weight_ok'] else 'no':>4s} "
            f"{'yes' if r['on_path'] else '':>7s}"
        )


def analyse_commute(cfg, g, points, landmarks, route, weekday: str) -> list[dict]:
    print(f"\n### {route.label}  [{route.id}]  weekday={weekday}")
    print(f"    departure {route.departure_time} from {route.origin}")
    events = commute_timeline(route)
    print("    timeline:")
    for ev in events:
        ride = f"  ride {ev['ride_min']:.0f} min" if ev["kind"] == "ride" else ""
        print(f"      [{ev['kind']:4s}] {ev['label']}{ride}")

    # two Dijkstra per walk leg
    boards = []
    leg_duration = {}  # (from,to) -> t_a_to_b
    for leg in route.walk_legs:
        board = score_leg(g, leg.from_id, leg.to_id, landmarks)
        boards.append(board)
        leg_duration[(leg.from_id, leg.to_id)] = board.t_a_to_b_min
        print(
            f"    leg {leg.from_id}→{leg.to_id}: t={board.t_a_to_b_min:.2f} min, "
            f"Dijkstra {board.dijkstra_from_a_s * 1000:.0f} + {board.dijkstra_to_b_s * 1000:.0f} ms"
        )

    # fill walk-leg durations into timeline for arrival offsets
    for ev in events:
        if ev["kind"] == "walk":
            ev["leg_duration_min"] = leg_duration.get((ev["from"], ev["to"]), 0.0)

    dep_min = int(route.departure_time.split(":")[0]) * 60 + int(route.departure_time.split(":")[1])
    service = cfg.service_time_min
    parcel_kg = cfg.example_parcel_weight_kg

    rows = []
    for p in points:
        best = best_leg_for_point(boards, p)
        if math.isinf(best.detour_min):
            # point is beyond the walking cutoff on every leg of this commute
            rows.append({
                "point_id": p.id,
                "leg": "",
                "t_a_to_p": math.inf,
                "t_p_to_b": math.inf,
                "t_a_to_b": best.t_a_to_b_min,
                "detour": math.inf,
                "arrival": "",
                "open": False,
                "open_reason": "unreachable within walking cutoff",
                "weight_ok": False,
                "weight_reason": "",
                "on_path": False,
                "commute_id": route.id,
                "weekday": weekday,
                "departure": route.departure_time,
            })
            continue
        offset = arrival_offset_min(events, best.leg_from, best.leg_to, best.t_a_to_p_min)
        arrive_min = dep_min + offset
        arrive_hhmm = minutes_to_hhmm(arrive_min)
        open_ok, open_reason = is_open_at(p, weekday, arrive_hhmm, service)
        wt_ok, wt_reason = weight_ok(p, parcel_kg)
        leg_label = f"{best.leg_from}→{best.leg_to}"
        rows.append({
            "point_id": p.id,
            "leg": leg_label,
            "t_a_to_p": best.t_a_to_p_min,
            "t_p_to_b": best.t_p_to_b_min,
            "t_a_to_b": best.t_a_to_b_min,
            "detour": best.detour_min,
            "arrival": arrive_hhmm,
            "open": open_ok,
            "open_reason": open_reason,
            "weight_ok": wt_ok,
            "weight_reason": wt_reason,
            "on_path": best.on_shortest_path,
            "commute_id": route.id,
            "weekday": weekday,
            "departure": route.departure_time,
        })
    rows.sort(key=lambda r: r["detour"] if not math.isinf(r["detour"]) else 1e9)
    return rows


def check_invariants(all_rows: list[dict]) -> None:
    print("\n===== Invariant checks =====")
    bad_neg = [r for r in all_rows if r["detour"] < -1e-6]
    print(f"  detour ≥ 0:  {len(all_rows) - len(bad_neg)}/{len(all_rows)} OK", end="")
    if bad_neg:
        print(f"  FAIL {bad_neg[:3]}")
    else:
        print("  PASS")

    on_path = [r for r in all_rows if r["on_path"]]
    far_zero = [r for r in on_path if r["detour"] > 1e-3]
    print(f"  detour≈0 for on-path points: {len(on_path)} on-path, ", end="")
    if not on_path:
        print("n/a")
    elif far_zero:
        print(f"FAIL {far_zero[:3]}")
    else:
        print(f"all |detour|<0.001 min PASS")

    # best-leg detour should be <= detour on any single fixed leg — implicit via min()
    for r in all_rows:
        if not math.isinf(r["detour"]) and r["detour"] < -1e-9:
            print("  FAIL negative", r)
            break


def main():
    cfg = load_habitual_config()
    g = load_graph()
    points = load_pickup_points()
    snap_points(g, points)
    landmarks = {k: info["node"] for k, info in snap_landmarks(g).items()}

    print(f"Graph {g.number_of_nodes()} nodes / {g.number_of_edges()} edges")
    print(f"service_time_min={cfg.service_time_min}, parcel={cfg.example_parcel_weight_kg} kg")
    print(f"ride times: " + ", ".join(
        f"{t.mode}:{t.ride_min:.0f}min"
        for r in cfg.routes for t in r.transfers
    ))

    all_rows: list[dict] = []
    by_commute = {}
    for route in cfg.routes:
        # demo: one morning + one evening
        if route.id not in ("morning_tram_metro", "evening_metro_only"):
            continue
        weekday = "wed"
        rows = analyse_commute(cfg, g, points, landmarks, route, weekday)
        by_commute[route.id] = rows
        all_rows.extend(rows)
        format_table(rows, f"{route.label} (weekday={weekday}, dep {route.departure_time})")

    check_invariants(all_rows)

    # save
    import pandas as pd

    pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")

    # summary counts
    print("\n===== Summary =====")
    for cid, rows in by_commute.items():
        open_n = sum(r["open"] for r in rows)
        wt_n = sum(r["weight_ok"] for r in rows)
        both = sum(r["open"] and r["weight_ok"] for r in rows)
        n = len(rows)
        print(f"  {cid}: open {open_n}/{n}, weight {wt_n}/{n}, both {both}/{n}")
        print(f"    min detour = {min(r['detour'] for r in rows):.2f} min "
              f"({min(rows, key=lambda r: r['detour'])['point_id']})")


if __name__ == "__main__":
    main()
