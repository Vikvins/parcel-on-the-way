#!/usr/bin/env python3
"""Step 7: experiment with 200 synthetic users + real commute case.

Methods (all scored with the same extra-walking metric vs habitual legs):
  (a) baseline_home_straight — closest to home, straight-line distance
  (b) baseline_home_walk     — closest to home, walking-network distance
  (c) ours                   — smallest usable detour across the user's commutes

Extra metric (Ma et al., 2024): days from parcel arrival (random weekday)
until the first usable pickup on the user's commutes (0 = same day).
Reports mean/median and share within 1 day and within the 7-day storage period.

Pickup points CSV is interchangeable. Fictional points = pipeline check only.
"""

from __future__ import annotations

import json
import math
import sys
import time as time_mod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.delay import (  # noqa: E402
    STORAGE_DAYS_DEFAULT,
    days_to_first_usable,
    summarize_delays,
    weekday_after,
)
from parcel_on_the_way.detour import (  # noqa: E402
    detour_for_point,
    score_leg,
)
from parcel_on_the_way.graph import dijkstra_with_prev, load_graph  # noqa: E402
from parcel_on_the_way.hours import hhmm_to_minutes, is_open_at, minutes_to_hhmm, weight_ok  # noqa: E402
from parcel_on_the_way.points import (  # noqa: E402
    WEEKDAYS,
    load_pickup_points,
    snap_points,
)
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402
from parcel_on_the_way.scenarios import DetourEngine, weekday_of  # noqa: E402

OUT_DIR = ROOT / "data" / "processed"
N_SYNTHETIC = 200
SEED = 42
MIN_HOME_DEST_M = 1500.0
METRO_SPEED_M_PER_MIN = 40_000 / 60  # 40 km/h
RIDE_WAIT_MIN = 3.0
STORAGE_DAYS = STORAGE_DAYS_DEFAULT

# Known transit stops (graph.LANDMARKS keys) used as synthetic "stations".
STOP_KEYS = (
    "tram_60letiya",
    "metro_leninsky_south",
    "oktyabrskaya_exit",
    "tram_oktyabrskaya",
    "tram_udarnitsa",
)

METHODS = ("baseline_home_straight", "baseline_home_walk", "ours")


# ---------------------------------------------------------------------------
# Synthetic users
# ---------------------------------------------------------------------------

@dataclass
class CommuteLeg:
    from_node: int
    to_node: int
    label: str


@dataclass
class SynthCommute:
    period: str  # morning | evening
    departure: str  # hhmm
    walk_legs: list[CommuteLeg]
    ride_min: float
    weekdays: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri")


@dataclass
class SynthUser:
    user_id: str
    home_node: int
    dest_node: int
    home_xy: tuple[float, float]
    dest_xy: tuple[float, float]
    home_stop: int
    dest_stop: int
    morning: SynthCommute
    evening: SynthCommute
    boards: dict = field(default_factory=dict)  # (from,to) -> LegScoreBoard

    def active_commutes(self, weekday: str) -> list[SynthCommute]:
        return [c for c in (self.morning, self.evening) if weekday in c.weekdays]


def _xy(graph, node: int) -> tuple[float, float]:
    return float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])


def _dist_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ride_min(a_xy: tuple[float, float], b_xy: tuple[float, float]) -> float:
    return RIDE_WAIT_MIN + _dist_m(a_xy, b_xy) / METRO_SPEED_M_PER_MIN


def generate_synthetic_users(graph, landmark_nodes: dict[str, int], n: int, rng) -> list[SynthUser]:
    nodes = list(graph.nodes)
    stop_nodes = [landmark_nodes[k] for k in STOP_KEYS if k in landmark_nodes]
    if len(stop_nodes) < 2:
        raise RuntimeError("Need at least two transit stops in landmark_nodes")

    users: list[SynthUser] = []
    attempts = 0
    while len(users) < n and attempts < n * 20:
        attempts += 1
        home_node = int(rng.choice(nodes))
        dest_node = int(rng.choice(nodes))
        hxy, dxy = _xy(graph, home_node), _xy(graph, dest_node)
        if _dist_m(hxy, dxy) < MIN_HOME_DEST_M:
            continue

        home_stop = min(stop_nodes, key=lambda s: _dist_m(hxy, _xy(graph, s)))
        dest_stop = min(stop_nodes, key=lambda s: _dist_m(dxy, _xy(graph, s)))
        if home_stop == dest_stop:
            # pick the other stop that is closer to dest than home_stop
            others = [s for s in stop_nodes if s != home_stop]
            dest_stop = min(others, key=lambda s: _dist_m(dxy, _xy(graph, s)))

        ride = _ride_min(_xy(graph, home_stop), _xy(graph, dest_stop))
        m_start, m_end = 8 * 60, 10 * 60 + 30  # 08:00–10:30
        e_start, e_end = 16 * 60, 21 * 60
        morning = SynthCommute(
            period="morning",
            departure=minutes_to_hhmm(int(rng.integers(m_start, m_end + 1))),
            walk_legs=[
                CommuteLeg(home_node, home_stop, "дом → остановка"),
                CommuteLeg(dest_stop, dest_node, "остановка → назначение"),
            ],
            ride_min=ride,
        )
        evening = SynthCommute(
            period="evening",
            departure=minutes_to_hhmm(int(rng.integers(e_start, e_end + 1))),
            walk_legs=[
                CommuteLeg(dest_node, dest_stop, "назначение → остановка"),
                CommuteLeg(home_stop, home_node, "остановка → дом"),
            ],
            ride_min=ride,
        )
        user = SynthUser(
                user_id=f"U{len(users):03d}",
                home_node=home_node,
                dest_node=dest_node,
                home_xy=hxy,
                dest_xy=dxy,
                home_stop=home_stop,
                dest_stop=dest_stop,
                morning=morning,
                evening=evening,
            )
        # skip users whose walk legs have no path within the walking cutoff
        # (the OSM walk graph is only weakly connected)
        try:
            for commute in (morning, evening):
                for leg in commute.walk_legs:
                    _ensure_board(user, leg, graph)
        except nx.NetworkXNoPath:
            continue
        users.append(user)
    if len(users) < n:
        raise RuntimeError(f"Only generated {len(users)}/{n} synthetic users")
    return users


# ---------------------------------------------------------------------------
# Scoring one point on one synthetic commute
# ---------------------------------------------------------------------------

def _ensure_board(user: SynthUser, leg: CommuteLeg, graph) -> "object":
    key = (leg.from_node, leg.to_node)
    if key not in user.boards:
        user.boards[key] = score_leg(
            graph,
            f"n{leg.from_node}",
            f"n{leg.to_node}",
            {f"n{leg.from_node}": leg.from_node, f"n{leg.to_node}": leg.to_node},
        )
    return user.boards[key]


def eval_synth_point(
    user: SynthUser,
    commute: SynthCommute,
    point,
    weekday: str,
    graph,
    service_min: float,
    parcel_weight_kg: float,
) -> dict | None:
    """Score a point on one synthetic commute. None if unreachable."""
    scored = []
    for leg in commute.walk_legs:
        board = _ensure_board(user, leg, graph)
        scored.append((leg, board, detour_for_point(board, point)))
    scored = [(leg, b, r) for leg, b, r in scored if not math.isinf(r.detour_min)]
    if not scored:
        return None
    leg, board, best = min(scored, key=lambda t: t[2].detour_min)

    # arrival offset: full earlier walk legs + ride + t(A→P)
    events = [
        {
            "kind": "walk",
            "from": f"n{L.from_node}",
            "to": f"n{L.to_node}",
            "label": L.label,
            "leg_duration_min": _ensure_board(user, L, graph).t_a_to_b_min,
            "ride_min": 0.0,
        }
        for L in commute.walk_legs[: commute.walk_legs.index(leg)]
    ]
    # ride is between the two walk legs (home_stop → dest_stop or reverse)
    if len(commute.walk_legs) == 2:
        events.append(
            {
                "kind": "ride",
                "from": "ride",
                "to": "ride",
                "label": "transport",
                "ride_min": commute.ride_min,
                "leg_duration_min": 0.0,
            }
        )
    events.append(
        {
            "kind": "walk",
            "from": f"n{leg.from_node}",
            "to": f"n{leg.to_node}",
            "label": leg.label,
            "leg_duration_min": board.t_a_to_b_min,
            "ride_min": 0.0,
        }
    )
    # rebuild timeline offset manually
    offset = 0.0
    for ev in events:
        if ev["kind"] == "walk" and ev["from"] == f"n{leg.from_node}" and ev["to"] == f"n{leg.to_node}":
            offset += best.t_a_to_p_min
            break
        if ev["kind"] == "walk":
            offset += ev.get("leg_duration_min", 0.0)
        else:
            offset += ev.get("ride_min", 0.0)

    dep = hhmm_to_minutes(commute.departure)
    arrive_min = dep + offset
    arrive_hhmm = minutes_to_hhmm(arrive_min)
    open_ok, open_reason = is_open_at(point, weekday, arrive_hhmm, service_min)
    wt_ok, wt_reason = weight_ok(point, parcel_weight_kg)

    return {
        "point_id": point.id,
        "period": commute.period,
        "detour_min": round(best.detour_min, 2),
        "on_shortest_path": best.on_shortest_path,
        "arrival": arrive_hhmm,
        "arrival_minutes": arrive_min,
        "open": open_ok,
        "open_reason": open_reason,
        "weight_ok": wt_ok,
        "usable": open_ok and wt_ok,
    }


# ---------------------------------------------------------------------------
# Real-user commute evaluation (reuse DetourEngine)
# ---------------------------------------------------------------------------

def eval_real_point(engine: DetourEngine, route, point, weekday, day, weight):
    from parcel_on_the_way.scenarios import _evaluate_point_on_commute

    return _evaluate_point_on_commute(
        engine, route, point, weekday, day, weight
    )


# ---------------------------------------------------------------------------
# Method selection
# ---------------------------------------------------------------------------

def pick_baseline_straight(points, home_xy: tuple[float, float], graph):
    best = None
    best_d = math.inf
    for p in points:
        if p.node is None:
            continue
        d = _dist_m(home_xy, _xy(graph, p.node))
        if d < best_d:
            best_d = d
            best = p
    return best


def pick_baseline_walk(points, home_dist: dict[int, float]):
    best = None
    best_d = math.inf
    for p in points:
        if p.node is None:
            continue
        d = home_dist.get(p.node, math.inf)
        if d < best_d:
            best_d = d
            best = p
    return best


def pick_ours(
    points,
    user: SynthUser,
    graph,
    weekday: str,
    service_min: float,
    weight: float,
):
    """Smallest detour among points usable on at least one commute that day."""
    best = None
    best_key = (math.inf, math.inf)
    for p in points:
        usable = []
        any_reachable = []
        for c in user.active_commutes(weekday):
            ev = eval_synth_point(user, c, p, weekday, graph, service_min, weight)
            if ev is None:
                continue
            any_reachable.append(ev)
            if ev["usable"]:
                usable.append(ev)
        if usable:
            key = min((e["detour_min"], e["arrival_minutes"]) for e in usable)
        elif any_reachable:
            # keep as weak candidate so we still pick something
            key = (min(e["detour_min"] for e in any_reachable) + 1000.0, 0)
        else:
            continue
        if key < best_key:
            best_key = key
            best = p
    return best


def usable_on_weekday(
    point,
    commutes_for_wd: list,
    weekday: str,
    graph,
    service_min: float,
    weight: float,
    user: SynthUser,
) -> bool:
    for c in commutes_for_wd:
        ev = eval_synth_point(user, c, point, weekday, graph, service_min, weight)
        if ev is not None and ev["usable"]:
            return True
    return False


# ---------------------------------------------------------------------------
# Delay metric for a chosen point
# ---------------------------------------------------------------------------

def delay_for_synth_point(user: SynthUser, point, graph, service_min, weight, arrival_weekday):
    def usable(wd: str) -> bool:
        return usable_on_weekday(
            point, user.active_commutes(wd), wd, graph, service_min, weight, user
        )

    return days_to_first_usable(arrival_weekday, usable, storage_days=STORAGE_DAYS)


def delay_if_separate_trip(point, arrival_weekday: str):
    """Days until the point is open on some calendar day (dedicated trip from home).

    Complements the commute-only metric: baselines that pick a point near home
    can often collect it on a separate walk, without waiting for a commute.
    """

    def open_someday(wd: str) -> bool:
        return point.is_open_on(wd)

    return days_to_first_usable(arrival_weekday, open_someday, storage_days=STORAGE_DAYS)


def delay_for_real_point(engine, point, weight, arrival_weekday, day0: date):
    def usable(wd: str) -> bool:
        # map weekday onto the next calendar day that matches wd, relative to day0
        # The user's routes only run mon–fridays; hours change with weekday.
        for route in engine.cfg.routes:
            if wd not in route.weekdays:
                continue
            # date with this weekday at least day0
            d = day0
            for _ in range(7):
                if weekday_of(d) == wd:
                    break
                d = date.fromordinal(d.toordinal() + 1)
            ev = eval_real_point(
                engine, route, point, weekday_of(d), d, weight
            )
            if ev is not None and ev.get("usable"):
                return True
        return False

    return days_to_first_usable(arrival_weekday, usable, storage_days=STORAGE_DAYS)


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def main():
    t0 = time_mod.perf_counter()
    rng = np.random.default_rng(SEED)

    print("Loading graph…")
    graph = load_graph()
    print(f"  nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    cfg = load_habitual_config()
    points = snap_points(graph, load_pickup_points())
    print(f"  pickup points: {len(points)}  service={cfg.service_time_min} min  "
          f"weight={cfg.example_parcel_weight_kg} kg")

    # landmark nodes for real + synthetic stops
    from parcel_on_the_way.points import snap_landmarks

    snaps = snap_landmarks(graph)
    landmark_nodes = {k: v["node"] for k, v in snaps.items()}

    print(f"\nGenerating {N_SYNTHETIC} synthetic users (seed={SEED})…")
    users = generate_synthetic_users(graph, landmark_nodes, N_SYNTHETIC, rng)
    print(f"  done in {time_mod.perf_counter() - t0:.1f}s")

    weight = cfg.example_parcel_weight_kg
    service_min = cfg.service_time_min

    # warm-up home Dijkstras are done per user inside baseline_walk
    rows = []
    delay_by_method: dict[str, list] = {m: [] for m in METHODS}
    sep_by_method: dict[str, list] = {m: [] for m in METHODS}
    examples = []

    print("\n=== Synthetic users ===")
    for i, user in enumerate(users):
        if i % 20 == 0:
            print(f"  … {i}/{len(users)}  ({time_mod.perf_counter() - t0:.1f}s)")

        # random arrival weekday Mon–Fri
        arrival_wd = WEEKDAYS[int(rng.integers(0, 5))]
        # random parcel weight around the default
        pw = float(rng.choice([0.5, 1.0, 2.0, 5.0, 10.0]))

        home_dist, _ = dijkstra_with_prev(graph, user.home_node)

        chosen = {
            "baseline_home_straight": pick_baseline_straight(points, user.home_xy, graph),
            "baseline_home_walk": pick_baseline_walk(points, home_dist),
            "ours": pick_ours(points, user, graph, arrival_wd, service_min, pw),
        }

        for method, pt in chosen.items():
            if pt is None:
                delay_by_method[method].append(None)
                sep_by_method[method].append(None)
                rows.append({
                    "user_id": user.user_id,
                    "method": method,
                    "point_id": None,
                    "arrival_weekday": arrival_wd,
                    "parcel_weight_kg": pw,
                    "delay_days": None,
                    "within_1_day": False,
                    "within_storage": False,
                    "delay_days_separate_trip": None,
                    "home_walk_min": None,
                    "detour_min": None,
                })
                continue

            # home walk for reporting
            hw = home_dist.get(pt.node, math.inf)
            hw = None if math.isinf(hw) else round(hw, 2)

            # best detour that day for reporting
            best_det = None
            for c in user.active_commutes(arrival_wd):
                ev = eval_synth_point(user, c, pt, arrival_wd, graph, service_min, pw)
                if ev is not None and (best_det is None or ev["detour_min"] < best_det):
                    best_det = ev["detour_min"]

            dres = delay_for_synth_point(user, pt, graph, service_min, pw, arrival_wd)
            dsep = delay_if_separate_trip(pt, arrival_wd)
            delay_by_method[method].append(dres)
            sep_by_method.setdefault(method, []).append(dsep)
            row = {
                "user_id": user.user_id,
                "method": method,
                "point_id": pt.id,
                "point_name": pt.name,
                "arrival_weekday": arrival_wd,
                "parcel_weight_kg": pw,
                "delay_days": dres.delay_days,
                "within_1_day": dres.within_1_day,
                "within_storage": dres.within_storage,
                "first_opportunity_weekday": dres.first_opportunity_weekday,
                "delay_days_separate_trip": dsep.delay_days,
                "home_walk_min": hw,
                "detour_min": best_det,
            }
            rows.append(row)
            if len(examples) < 3 and method == "ours" and i < 30:
                examples.append({**row, "user_id": user.user_id, "home_node": user.home_node})

    # ---- aggregates ----
    summary = {}
    for method in METHODS:
        summary[method] = {
            "delay": summarize_delays(delay_by_method[method], storage_days=STORAGE_DAYS),
            "delay_separate_trip": summarize_delays(sep_by_method[method], storage_days=STORAGE_DAYS),
        }
        # extra-time stats among rows with numeric detour
        detours = [
            r["detour_min"] for r in rows
            if r["method"] == method and r.get("detour_min") is not None
        ]
        homes = [
            r["home_walk_min"] for r in rows
            if r["method"] == method and r.get("home_walk_min") is not None
        ]
        summary[method]["mean_detour_min"] = float(np.mean(detours)) if detours else None
        summary[method]["median_detour_min"] = float(np.median(detours)) if detours else None
        summary[method]["share_detour_le_5"] = (
            float(np.mean([d <= 5.0 for d in detours])) if detours else 0.0
        )
        summary[method]["mean_home_walk_min"] = float(np.mean(homes)) if homes else None

    # ours vs baselines (share where ours has lower delay / detour)
    by_user = {}
    for r in rows:
        by_user.setdefault(r["user_id"], {})[r["method"]] = r
    wins_delay = {"vs_straight": 0, "vs_walk": 0, "n": 0}
    wins_detour = {"vs_straight": 0, "vs_walk": 0, "n": 0}
    for uid, m in by_user.items():
        o = m.get("ours")
        if not o:
            continue
        wins_delay["n"] += 1
        wins_detour["n"] += 1
        for key, base in (("vs_straight", "baseline_home_straight"), ("vs_walk", "baseline_home_walk")):
            b = m.get(base)
            if b is None:
                continue
            od, bd = o.get("delay_days"), b.get("delay_days")
            if od is not None and bd is not None and od < bd:
                wins_delay[key] += 1
            elif od is not None and bd is not None and od == bd:
                wins_delay[key] += 0  # tie
            odet, bdet = o.get("detour_min"), b.get("detour_min")
            if odet is not None and bdet is not None and odet < bdet - 1e-9:
                wins_detour[key] += 1

    # ---- real user case ----
    print("\n=== Real user (habitual routes) ===")
    engine = DetourEngine(graph, landmark_nodes, cfg)
    real_day = date(2026, 10, 14)  # Wednesday
    real_wd = weekday_of(real_day)
    real_rows = []
    real_delays = {m: [] for m in METHODS}

    # pick points with three strategies on the real day
    real_candidates = []
    for p in points:
        opts = []
        for route in cfg.routes:
            if real_wd not in route.weekdays:
                continue
            ev = eval_real_point(engine, route, p, real_wd, real_day, weight)
            if ev is not None:
                opts.append(ev)
        if opts:
            usable = [e for e in opts if e["usable"]]
            real_candidates.append((p, usable or opts))

    if real_candidates:
        # ours: min detour among usable
        ours_real = None
        best = math.inf
        for p, opts in real_candidates:
            usable = [e for e in opts if e["usable"]]
            if not usable:
                continue
            d = min(e["detour_min"] for e in usable)
            if d < best:
                best = d
                ours_real = p
        # baselines
        home_xy = _xy(graph, landmark_nodes["home"])
        base_s = pick_baseline_straight(points, home_xy, graph)
        home_dist_r, _ = dijkstra_with_prev(graph, landmark_nodes["home"])
        base_w = pick_baseline_walk(points, home_dist_r)

        for method, pt in (
            ("baseline_home_straight", base_s),
            ("baseline_home_walk", base_w),
            ("ours", ours_real),
        ):
            if pt is None:
                continue
            dres = delay_for_real_point(engine, pt, weight, real_wd, real_day)
            real_delays[method].append(dres)
            real_rows.append({
                "method": method,
                "point_id": pt.id,
                "point_name": pt.name,
                "arrival_weekday": real_wd,
                "delay_days": dres.delay_days,
                "within_1_day": dres.within_1_day,
                "within_storage": dres.within_storage,
            })
            print(f"  {method:28s} → {pt.id:18s} delay={dres.delay_days} d")

    real_summary = {m: summarize_delays(real_delays[m], storage_days=STORAGE_DAYS) for m in METHODS}

    # ---- write CSV ----
    import csv

    csv_path = OUT_DIR / "step7_experiment.csv"
    fields = [
        "user_id", "method", "point_id", "point_name", "arrival_weekday",
        "parcel_weight_kg", "delay_days", "within_1_day", "within_storage",
        "first_opportunity_weekday", "delay_days_separate_trip",
        "home_walk_min", "detour_min",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nCSV → {csv_path}")

    json_path = OUT_DIR / "step7_summary.json"
    payload = {
        "seed": SEED,
        "n_synthetic": N_SYNTHETIC,
        "storage_days": STORAGE_DAYS,
        "service_time_min": service_min,
        "methods": summary,
        "wins_delay": wins_delay,
        "wins_detour": wins_detour,
        "real_user": {"date": real_day.isoformat(), "rows": real_rows, "summary": real_summary},
        "note": "Pickup points are fictional — pipeline check only.",
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"JSON → {json_path}")

    # ---- printed table ----
    print("\n=== Delay days until first usable pickup on commutes ===")
    print(f"{'method':28s} {'mean':>6s} {'median':>7s} {'≤1 day':>8s} {'≤7 days':>8s} {'n':>5s}")
    for method in METHODS:
        d = summary[method]["delay"]
        mean_s = f"{d['mean_delay_days']:.2f}" if d["mean_delay_days"] is not None else "—"
        med_s = f"{d['median_delay_days']:.1f}" if d["median_delay_days"] is not None else "—"
        print(
            f"{method:28s} {mean_s:>6s} {med_s:>7s} "
            f"{d['share_within_1_day']:7.1%} {d['share_within_storage']:7.1%} {d['n']:5d}"
        )

    print("\n=== Delay days if a dedicated trip is allowed (open-day check) ===")
    print(f"{'method':28s} {'mean':>6s} {'median':>7s} {'≤1 day':>8s} {'≤7 days':>8s} {'n':>5s}")
    for method in METHODS:
        d = summary[method]["delay_separate_trip"]
        mean_s = f"{d['mean_delay_days']:.2f}" if d["mean_delay_days"] is not None else "—"
        med_s = f"{d['median_delay_days']:.1f}" if d["median_delay_days"] is not None else "—"
        print(
            f"{method:28s} {mean_s:>6s} {med_s:>7s} "
            f"{d['share_within_1_day']:7.1%} {d['share_within_storage']:7.1%} {d['n']:5d}"
        )

    print("\n=== Extra walking (detour vs habitual legs) ===")
    print(f"{'method':28s} {'mean':>7s} {'median':>7s} {'≤5 min':>8s}")
    for method in METHODS:
        s = summary[method]
        mean_s = f"{s['mean_detour_min']:.2f}" if s["mean_detour_min"] is not None else "—"
        med_s = f"{s['median_detour_min']:.2f}" if s["median_detour_min"] is not None else "—"
        print(
            f"{method:28s} {mean_s:>7s} {med_s:>7s} {s['share_detour_le_5']:7.1%}"
        )

    print("\n=== ours better (strictly lower) ===")
    print(f"  delay:  vs straight {wins_delay['vs_straight']}/{wins_delay['n']}, "
          f"vs walk {wins_delay['vs_walk']}/{wins_delay['n']}")
    print(f"  detour: vs straight {wins_detour['vs_straight']}/{wins_detour['n']}, "
          f"vs walk {wins_detour['vs_walk']}/{wins_detour['n']}")

    # ---- charts ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # chart 1: delay distribution
    ax = axes[0]
    bins = np.arange(-0.5, 9.5, 1.0)
    width = 0.25
    for i, method in enumerate(METHODS):
        delays = [r.delay_days for r in delay_by_method[method] if r and r.delay_days is not None]
        if not delays:
            continue
        hist, _ = np.histogram(delays, bins=bins)
        ax.bar(bins[:-1] + i * width, hist, width=width, label=method, alpha=0.85)
    ax.set_xlabel("дни до первого удобного получения")
    ax.set_ylabel("число пользователей")
    ax.set_title("Задержка получения (коммьюты)")
    ax.legend(fontsize=8)
    ax.set_xticks(range(0, 9))

    # chart 2: share within windows
    ax = axes[1]
    x = np.arange(len(METHODS))
    w1 = [summary[m]["delay"]["share_within_1_day"] for m in METHODS]
    w7 = [summary[m]["delay"]["share_within_storage"] for m in METHODS]
    ax.bar(x - 0.15, w1, width=0.3, label="≤ 1 дня", color="#2a9d8f")
    ax.bar(x + 0.15, w7, width=0.3, label="≤ 7 дней (хранение)", color="#e9c46a")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("baseline_home_", "base_") for m in METHODS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("доля пользователей")
    ax.set_title("Успевают забрать вовремя")
    ax.legend(fontsize=8)

    fig.tight_layout()
    chart_path = OUT_DIR / "step7_charts.png"
    fig.savefig(chart_path, dpi=140)
    print(f"\nCharts → {chart_path}")

    # example table
    if examples:
        print("\n=== Примеры (ours) ===")
        for ex in examples:
            print(
                f"  {ex['user_id']}: {ex['point_id']} "
                f"arrival={ex['arrival_weekday']} delay={ex['delay_days']} d "
                f"detour={ex['detour_min']} min home_walk={ex['home_walk_min']} min"
            )

    print(f"\nTotal {time_mod.perf_counter() - t0:.1f}s")
    print("Note: pickup points are fictional — pipeline check only.")


if __name__ == "__main__":
    main()
