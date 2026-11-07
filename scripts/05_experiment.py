#!/usr/bin/env python3
"""Step 7: experiment — 200 synthetic users + real-user case, 3 methods.

Shared cost for EVERY method and EVERY point:
    cost(point) = min(best usable detour on the user's commutes,
                      round trip home → point → home at a usable time)
    ours = argmin cost

  PYTHONPATH=src .venv/bin/python scripts/05_experiment.py \
      --csv data/pickup_points.csv --n 200 --seed 20261015

Results on the 10 fictional points are a PIPELINE CHECK only.
Pass a real CDEK CSV (see data/cdek_points_template.csv) for real results.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.experiment import (  # noqa: E402
    SEED,
    UserEngine,
    days_until_on_the_way,
    days_until_pickup,
    hours_removal_stats,
    load_transit_stops,
    make_synthetic_user,
    score_method,
    snap_stops,
)
from parcel_on_the_way.graph import load_graph  # noqa: E402
from parcel_on_the_way.points import (  # noqa: E402
    load_any_points,
    snap_landmarks,
    snap_points,
)
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402
from parcel_on_the_way.scenarios import (  # noqa: E402
    DetourEngine,
    scenario1_pick_point,
)

OUT_DIR = ROOT / "data" / "results"
METHODS = ("baseline_straight", "baseline_network", "ours")
METHOD_RU = {
    "baseline_straight": "baseline: ближайший к дому по прямой",
    "baseline_network": "baseline: ближайший к дому по сети",
    "ours": "наш метод: min cost",
}
EPS = 1e-9


def compare_ours_vs(user_rows: list[dict], other: str) -> dict[str, float]:
    better = equal = worse = 0
    for r in user_rows:
        a = r["ours"]["extra_min"]
        b = r[other]["extra_min"]
        both_inf = (not math.isfinite(a)) and (not math.isfinite(b))
        if both_inf:
            equal += 1
        elif math.isfinite(a) and not math.isfinite(b):
            better += 1
        elif not math.isfinite(a) and math.isfinite(b):
            worse += 1
        elif a < b - EPS:
            better += 1
        elif a > b + EPS:
            worse += 1
        else:
            equal += 1
    n = max(1, len(user_rows))
    return {
        "better": better / n,
        "equal": equal / n,
        "worse": worse / n,
        "worse_count": worse,
        "n": len(user_rows),
    }


def summarise(user_rows: list[dict], hours_morning: dict, hours_evening: dict,
              days_rows: list[dict]) -> pd.DataFrame:
    rows = []
    # users where ALL three methods are usable (for mean/median, no caps)
    common = [
        r for r in user_rows
        if all(math.isfinite(r[m]["extra_min"]) for m in METHODS)
    ]
    for method in METHODS:
        all_finite = [r[method]["extra_min"] for r in user_rows if math.isfinite(r[method]["extra_min"])]
        common_vals = [r[method]["extra_min"] for r in common]
        arr_all = np.array(all_finite) if all_finite else np.array([math.nan])
        arr_com = np.array(common_vals) if common_vals else np.array([math.nan])
        within5 = sum(
            1 for r in user_rows
            if math.isfinite(r[method]["extra_min"]) and r[method]["extra_min"] <= 5.0
        )
        # days to pickup
        dvals = [d[method] for d in days_rows if d[method] is not None]
        d_arr = np.array(dvals) if dvals else np.array([math.nan])
        within1 = sum(1 for d in dvals if d <= 1)
        within7 = len(dvals)

        rows.append({
            "method": method,
            "method_ru": METHOD_RU[method],
            "n_users": len(user_rows),
            "usable_share": sum(1 for r in user_rows if r[method]["usable"]) / len(user_rows),
            "n_common_all_methods_usable": len(common),
            "mean_extra_min_common": float(np.mean(arr_com)) if common_vals else math.nan,
            "median_extra_min_common": float(np.median(arr_com)) if common_vals else math.nan,
            "mean_extra_min_finite": float(np.mean(arr_all)) if all_finite else math.nan,
            "median_extra_min_finite": float(np.median(arr_all)) if all_finite else math.nan,
            "share_within_5min": within5 / len(user_rows),
            "mean_days_to_pickup": float(np.mean(d_arr)) if dvals else math.nan,
            "median_days_to_pickup": float(np.median(d_arr)) if dvals else math.nan,
            "share_days_le_1": within1 / len(days_rows) if days_rows else math.nan,
            "share_days_le_7": within7 / len(days_rows) if days_rows else math.nan,
            # delay of pickup ON THE WAY (detour ≤ 5 min), separate trips excluded
            "mean_days_on_way": float(np.mean(ow)) if (ow := [
                d[f"{method}_on_way"] for d in days_rows
                if d.get(f"{method}_on_way") is not None]) else math.nan,
            "share_on_way_le_1": (sum(1 for x in ow if x <= 1) / len(days_rows))
            if days_rows else math.nan,
            "share_on_way_le_7": (len(ow) / len(days_rows)) if days_rows else math.nan,
        })
    df = pd.DataFrame(rows)

    cmp_str = compare_ours_vs(user_rows, "baseline_straight")
    cmp_net = compare_ours_vs(user_rows, "baseline_network")
    print("\n===== Comparison sanity (ours vs baselines, same cost) =====")
    for label, c in (("straight", cmp_str), ("network", cmp_net)):
        print(f"  vs {label:8s}: better {c['better']:.1%} | equal {c['equal']:.1%} "
              f"| worse {c['worse']:.1%}  (worse n={c['worse_count']})")
        if c["worse_count"] != 0:
            print(f"    FAIL: ours must never be worse (shared cost, argmin)")
    return df


def plot_extra_distribution(user_rows: list[dict], path: Path):
    fig, ax = plt.subplots(figsize=(9, 5))
    data, labels = [], []
    for method in METHODS:
        vals = [r[method]["extra_min"] for r in user_rows if math.isfinite(r[method]["extra_min"])]
        data.append(vals)
        labels.append(METHOD_RU[method].split(": ", 1)[1])
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_ylabel("лишнее время, мин (без обрезки, только finite)")
    ax.set_title("Распределение лишнего времени по методам")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_within_5min(df: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [METHOD_RU[m].split(": ", 1)[1] for m in df["method"]]
    ax.bar(labels, df["share_within_5min"] * 100, color=["#8aa4b8", "#5b8fa8", "#c45c26"])
    ax.set_ylabel("доля пользователей, %")
    ax.set_title("Доля с пригодным вариантом ≤ 5 мин")
    ax.set_ylim(0, 100)
    for i, v in enumerate(df["share_within_5min"] * 100):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def describe_user(idx: int, row: dict, user) -> str:
    lines = [
        f"Пользователь #{idx} ({user.user_id}), {user.weekday}, "
        f"утро dep {user.morning_dep} / вечер dep {user.evening_dep}"
    ]
    for method in METHODS:
        m = row[method]
        if not math.isfinite(m["extra_min"]):
            lines.append(f"  {METHOD_RU[method]}: нет пригодного варианта")
            continue
        kind = "крюк по пути" if m["via"] == "detour" else "отдельная поездка туда-обратно"
        lines.append(
            f"  {METHOD_RU[method]}: {m['point_id']} → {kind} {m['extra_min']:.1f} мин"
            + (f", приход {m.get('arrival')}" if m.get("arrival") else "")
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/pickup_points.csv")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--parcel-kg", type=float, default=2.0)
    ap.add_argument("--stops", default="data/transit_stops.csv")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_habitual_config()
    g = load_graph()
    points = load_any_points(args.csv)
    snap_points(g, points)
    stops = load_transit_stops(ROOT / args.stops)
    snap_stops(g, stops)

    print("=" * 72)
    fictional = all(p.id.startswith(("PVZ-TEST", "POSTAMAT-TEST")) for p in points)
    if fictional:
        print("PIPELINE CHECK on fictional points — NOT real CDEK data.")
        print("Replace --csv with a real CDEK export (same columns) for real results.")
    else:
        n_unknown = sum("unknown" in p.note for p in points)
        print(f"Real CDEK points from {args.csv} ({n_unknown} with assumed hours)")
    print("=" * 72)
    print(f"points={len(points)}  stops={len(stops)}  n_users={args.n}  seed={args.seed}")
    print(f"parcel={args.parcel_kg} kg  service={cfg.service_time_min} min")
    print("cost = min(usable detour, usable round-trip home→point→home); ours = argmin")

    rng = random.Random(args.seed)
    engine = UserEngine(g, cfg)

    users = []
    skipped = 0
    attempts = 0
    while len(users) < args.n and attempts < args.n * 20:
        attempts += 1
        u = make_synthetic_user(rng, g, stops, f"U{len(users):03d}", cfg)
        if u is None:
            continue
        if not engine.user_reachable(u):
            skipped += 1  # some walk leg has no path within the cutoff
            continue
        users.append(u)
    print(f"generated users: {len(users)} (skipped unreachable: {skipped})")

    ref_day = date(2026, 10, 14)
    user_rows = []
    days_rows = []
    hours_morning = {"total": 0, "removed_hours": 0}
    hours_evening = {"total": 0, "removed_hours": 0}

    by_id = {p.id: p for p in points}
    for user in users:
        row = {"user_id": user.user_id, "weekday": user.weekday}
        for method in METHODS:
            row[method] = score_method(
                method, engine, user, points, user.weekday, ref_day, args.parcel_kg
            )
        user_rows.append(row)

        # days until first usable pickup (arrival weekday = user's weekday)
        drow = {"user_id": user.user_id}
        for method in METHODS:
            drow[method] = days_until_pickup(
                engine, user, points, method, user.weekday, args.parcel_kg, 7
            )
            chosen = by_id.get(row[method]["point_id"])
            drow[f"{method}_on_way"] = days_until_on_the_way(
                engine, user, chosen, user.weekday, args.parcel_kg, 5.0, 7
            )
        days_rows.append(drow)

        for period, acc in (("morning", hours_morning), ("evening", hours_evening)):
            st = hours_removal_stats(
                engine, user, points, user.weekday, ref_day, args.parcel_kg, period
            )
            acc["total"] += st["total"]
            acc["removed_hours"] += st["removed_hours"]

    df = summarise(user_rows, hours_morning, hours_evening, days_rows)

    print("\n===== Results (mean/median only over users where ALL 3 methods usable) =====")
    cols = [
        "method_ru", "usable_share",
        "mean_extra_min_common", "median_extra_min_common",
        "share_within_5min",
        "mean_days_to_pickup", "median_days_to_pickup",
        "share_days_le_1", "share_days_le_7",
        "mean_days_on_way", "share_on_way_le_1", "share_on_way_le_7",
    ]
    print(df[cols].to_string(index=False))
    print(f"\n  n users total: {len(user_rows)}")
    print(f"  n users where ALL 3 methods usable (means/medians above): "
          f"{df.iloc[0]['n_common_all_methods_usable']}")
    print("  (users with no option are EXCLUDED from mean/median — no 60-min cap)")

    print(f"\n  options removed by hours · morning: "
          f"{hours_morning['removed_hours']}/{hours_morning['total']} "
          f"= {hours_morning['removed_hours']/max(1,hours_morning['total']):.1%}")
    print(f"  options removed by hours · evening: "
          f"{hours_evening['removed_hours']}/{hours_evening['total']} "
          f"= {hours_evening['removed_hours']/max(1,hours_evening['total']):.1%}")

    # save
    flat = []
    for r, d in zip(user_rows, days_rows):
        rec = {"user_id": r["user_id"], "weekday": r["weekday"]}
        for m in METHODS:
            rec[f"{m}_point"] = r[m]["point_id"]
            rec[f"{m}_extra_min"] = r[m]["extra_min"]
            rec[f"{m}_via"] = r[m]["via"]
            rec[f"{m}_usable"] = r[m]["usable"]
            rec[f"{m}_days_to_pickup"] = d[m]
            rec[f"{m}_days_on_way"] = d.get(f"{m}_on_way")
        flat.append(rec)
    raw_path = OUT_DIR / "experiment_user_results.csv"
    pd.DataFrame(flat).to_csv(raw_path, index=False)
    summary_path = OUT_DIR / "experiment_summary.csv"
    df.to_csv(summary_path, index=False)

    plot_extra_distribution(user_rows, OUT_DIR / "extra_time_by_method.png")
    plot_within_5min(df, OUT_DIR / "share_within_5min.png")

    print("\n===== 3 example users =====")
    for idx in (0, len(users) // 2, len(users) - 1):
        if 0 <= idx < len(user_rows):
            print(describe_user(idx, user_rows[idx], users[idx]))
            print()

    print("===== Real user (author commutes, Wednesday) =====")
    landmarks = {k: info["node"] for k, info in snap_landmarks(g).items()}
    detour_engine = DetourEngine(g, landmarks, cfg)
    s1 = scenario1_pick_point(
        detour_engine, points, day=ref_day, parcel_weight_kg=args.parcel_kg
    )
    print(f"candidates: {s1['n_candidates']}")
    for r in s1["results"]:
        print(f"  [{r['reason']}] {r['point_id']}: {r['reason_ru']}")
    (OUT_DIR / "real_user_scenario1_wed.json").write_text(
        json.dumps(s1, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSaved: {summary_path}, {raw_path}, charts")
    if fictional:
        print("NOTE: fictional points — pipeline check only.")


if __name__ == "__main__":
    main()
