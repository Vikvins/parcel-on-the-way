#!/usr/bin/env python3
"""Step 2: load habitual routes + 10 fictional pickup points, snap, plot."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    dijkstra_with_prev,
    load_graph,
    path_coordinates,
    reconstruct_path,
    snap_to_node,
)
from parcel_on_the_way.points import (  # noqa: E402
    load_pickup_points,
    snap_landmarks,
    snap_points,
)
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402

OUT = ROOT / "data" / "processed" / "pickup_points_map.png"


def main():
    cfg = load_habitual_config()
    print("=== Habitual routes config ===")
    print(f"  walk_speed_kmh = {cfg.walk_speed_kmh}")
    print(f"  service_time_min = {cfg.service_time_min}")
    print(f"  experiment morning range = {cfg.morning_range[0]}–{cfg.morning_range[1]}")
    print(f"  experiment evening range = {cfg.evening_range[0]}–{cfg.evening_range[1]}")
    print(f"  routes: {len(cfg.routes)}")
    for r in cfg.routes:
        legs = " | ".join(f"{leg.from_id}→{leg.to_id}" for leg in r.walk_legs)
        print(f"    [{r.period:7s}] {r.id:22s} dep={r.departure_time}  walk: {legs}")

    g = load_graph()
    print(f"\nGraph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    points = load_pickup_points()
    snap_points(g, points)
    landmarks = snap_landmarks(g)

    print("\n=== Pickup points (FICTIONAL — not real CDEK data) ===")
    print(f"  {'id':18s} {'type':9s} {'kg':>4s} {'snap_m':>7s} {'sat':>11s} {'note':30s} name")
    for p in points:
        sat = p.hours_on("sat")
        sat_s = "closed" if sat[0] is None else f"{sat[0]}-{sat[1]}"
        print(
            f"  {p.id:18s} {p.point_type:9s} {p.max_weight_kg:4.0f} "
            f"{(p.snap_dist_m or 0):7.1f} {sat_s:>11s} {p.note[:30]:30s} {p.name}"
        )
        if (p.snap_dist_m or 0) > 50:
            print(f"    WARN large snap distance")

    # short-hours check for evening filter
    print("\n=== Evening filter preview (weekday, arrive after 18:00) ===")
    for p in points:
        end = p.hours_on("fri")[1]
        flag = "CUT if arrive after close" if end and end <= "18:00" else "ok for late evening"
        print(f"  {p.id:18s} fri {p.hours_on('fri')[0]}-{end or '—':>5}  {flag}")

    # --- map ---
    print("\n=== Map ===")
    fig, ax = plt.subplots(figsize=(11, 11))
    for u, v, _k, data in g.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        c = list(geom.coords)
        ax.plot([p[0] for p in c], [p[1] for p in c], color="#e2e8ee",
                linewidth=0.3, zorder=1)

    # morning walk legs (as in config) in two colours
    leg_colors = ["#2f8f4e", "#1f6f8b", "#c45c26", "#6b4c9a"]
    seen = set()
    for route in cfg.routes:
        if route.period != "morning":
            continue
        for i, leg in enumerate(route.walk_legs):
            key = (leg.from_id, leg.to_id)
            if key in seen:
                continue
            seen.add(key)
            a = landmarks[leg.from_id]["node"]
            b = landmarks[leg.to_id]["node"]
            dist, prev = dijkstra_with_prev(g, a)
            path = reconstruct_path(prev, a, b)
            coords = path_coordinates(g, path)
            ax.plot(
                [c[0] for c in coords], [c[1] for c in coords],
                color=leg_colors[len(seen) % len(leg_colors)],
                linewidth=2.8, zorder=3, alpha=0.9,
                label=f"{leg.label} ({dist[b]:.1f} мин)",
            )

    # landmarks
    for key, info in landmarks.items():
        if key in ("tram_oktyabrskaya", "tram_udarnitsa"):
            ax.scatter([info["node_x"]], [info["node_y"]], s=40, color="#bbb",
                       marker="^", zorder=4)
            continue
        name = {
            "home": "Дом", "misis": "МИСИС",
            "tram_60letiya": "Трамвай\n«60-летия»",
            "metro_leninsky_south": "Метро\n«Ленинский» южн.",
            "oktyabrskaya_exit": "Метро\n«Октябрьская»",
        }.get(key, key)
        ax.scatter([info["node_x"]], [info["node_y"]], s=100, color="#1c2430",
                   marker="o", zorder=5, edgecolors="white")
        ax.annotate(name, (info["node_x"], info["node_y"]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=9, fontweight="bold", color="#1c2430")

    # pickup points
    for p in points:
        x, y = g.nodes[p.node]["x"], g.nodes[p.node]["y"]
        color = "#c45c26" if p.point_type == "pvz" else "#1f6f8b"
        marker = "s" if p.point_type == "pvz" else "D"
        # highlight short-hours (fri close <= 18:00) with edge ring
        fri_close = p.hours_on("fri")[1]
        short = fri_close is not None and fri_close <= "18:00"
        ax.scatter([x], [y], s=90, color=color, marker=marker, zorder=6,
                   edgecolors="#fff200" if short else "white", linewidths=2 if short else 1)
        label = f"{p.id.replace('PVZ-TEST-','П').replace('POSTAMAT-TEST-','ПМ')}"
        if short:
            label += "\n(до 18:00)"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, -12),
                    fontsize=7.5, color="#1c2430")

    ax.set_aspect("equal")
    ax.set_title(
        "Шаг 2: 10 вымышленных пунктов + пешие ноги привычных маршрутов\n"
        "ПВЗ ▭ / постамат ◇ · жёлтая обводка = закрывается ≤18:00 · НЕ реальные данные СДЭК",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")
    print(f"Points: {len(points)}, all snapped.")


if __name__ == "__main__":
    main()
