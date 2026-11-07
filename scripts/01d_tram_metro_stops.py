#!/usr/bin/env python3
"""Report confirmed commute stops/legs and draw the confirmation map."""

from __future__ import annotations

import math
import sys
import time as time_mod
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    HABITUAL_COMMUTES,
    HABITUAL_WALK_LEGS,
    LANDMARKS,
    LANDMARK_SOURCES,
    dijkstra_with_prev,
    load_graph,
    path_coordinates,
    reconstruct_path,
    snap_to_node,
)

OUT = ROOT / "data" / "processed" / "tram_metro_stops_map.png"

LABELS = {
    "home": "Дом",
    "misis": "МИСИС (Ленинский 4)",
    "tram_60letiya": "Трамвай «Проспект 60-летия Октября»",
    "metro_leninsky_south": "Метро «Ленинский проспект» (южный вход)",
    "tram_oktyabrskaya": "Трамвай «Метро «Октябрьская»» (справочно)",
    "oktyabrskaya_exit": "Метро «Октябрьская» (уличный выход)",
    "tram_udarnitsa": "Трамвай «Фабрика «Ударница»» (справочно)",
}


def main():
    g = load_graph()
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    print("\n=== 1) Stops / landmarks ===")
    snaps = {}
    print(f"  {'name':50s} {'lon':>10s} {'lat':>9s} {'snap_m':>7s}")
    for key, lon_lat in LANDMARKS.items():
        info = snap_to_node(g, lon_lat)
        snaps[key] = info
        print(
            f"  {LABELS[key][:50]:50s} "
            f"{lon_lat[0]:10.6f} {lon_lat[1]:9.6f} {info['snap_dist_m']:7.1f}"
        )
        print(f"      {LANDMARK_SOURCES[key]}")

    print("\n=== 2) Walking legs (tram/metro rides are not routed) ===")
    print(f"  {'leg':52s} {'t_min':>7s} {'straight_m':>11s} {'Dij_s':>7s}")
    legs = {}
    for name, a_key, b_key in HABITUAL_WALK_LEGS:
        a = snaps[a_key]["node"]
        b = snaps[b_key]["node"]
        t0 = time_mod.perf_counter()
        dist, prev = dijkstra_with_prev(g, a)
        dt = time_mod.perf_counter() - t0
        path = reconstruct_path(prev, a, b)
        tmin = dist[b]
        straight = math.hypot(
            g.nodes[b]["x"] - g.nodes[a]["x"],
            g.nodes[b]["y"] - g.nodes[a]["y"],
        )
        print(f"  {name:52s} {tmin:7.2f} {straight:11.0f} {dt:7.3f}")
        legs[name] = (path, tmin, a_key, b_key)

    print("\n=== 3) Commute options (for scenarios) ===")
    for commute in HABITUAL_COMMUTES:
        print(f"  [{commute['id']}] {commute['label']}:")
        for seg in commute["segments"]:
            walk = ""
            if seg["mode"] == "walk":
                name = next(n for n, (p, t, a, b) in legs.items()
                            if a == seg["from"] and b == seg["to"])
                walk = f"  → {legs[name][1]:.2f} мин пешком"
            print(f"    - {seg['label']}  [{seg['mode']}]{walk}")

    print("\n=== 4) Map ===")
    fig, ax = plt.subplots(figsize=(11, 10))
    for u, v, _k, data in g.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        c = list(geom.coords)
        ax.plot([p[0] for p in c], [p[1] for p in c], color="#d5dee6",
                linewidth=0.35, zorder=1)

    colors_leg = {
        "home → tram «Проспект 60-летия Октября»": "#2f8f4e",
        "home → metro «Ленинский проспект (южный)»": "#1f6f8b",
        "metro «Октябрьская» exit → MISIS": "#c45c26",
    }
    labels_leg = {
        "home → tram «Проспект 60-летия Октября»":
            "пешком к трамваю «60-летия Октября»",
        "home → metro «Ленинский проспект (южный)»":
            "пешком к м. «Ленинский проспект» (южн.)",
        "metro «Октябрьская» exit → MISIS":
            "пешком от м. «Октябрьская» до МИСИС",
    }
    for name, (path, tmin, a_key, b_key) in legs.items():
        coords = path_coordinates(g, path)
        ax.plot([c[0] for c in coords], [c[1] for c in coords],
                color=colors_leg[name], linewidth=2.8, zorder=3, alpha=0.92,
                label=f"{labels_leg[name]} ({tmin:.1f} мин)")

    # tram/metro transfer as dashed grey links (NOT walking routes)
    for a_key, b_key, lbl in (
        ("tram_60letiya", "metro_leninsky_south", "трамвай"),
        ("metro_leninsky_south", "oktyabrskaya_exit", "метро"),
    ):
        xa, ya = snaps[a_key]["node_x"], snaps[a_key]["node_y"]
        xb, yb = snaps[b_key]["node_x"], snaps[b_key]["node_y"]
        ax.plot([xa, xb], [ya, yb], color="#888", linewidth=1.6,
                linestyle="--", zorder=2, alpha=0.8,
                label=f"{lbl} (не пешком)")

    style = {
        "home": ("Дом", "#2f8f4e", "o", 110),
        "misis": ("МИСИС", "#c45c26", "s", 110),
        "tram_60letiya": ("Трамвай\n«60-летия Октября»", "#6b4c9a", "^", 95),
        "metro_leninsky_south": ("Метро «Ленинский\nпроспект» (южн.)", "#1f6f8b", "D", 95),
        "oktyabrskaya_exit": ("Метро «Октябрьская»\n(выход)", "#1f6f8b", "D", 95),
        "tram_oktyabrskaya": ("(трамвай, справочно)", "#bbb", "^", 60),
        "tram_udarnitsa": ("(трамвай, справочно)", "#bbb", "^", 60),
    }
    for key, (label, color, marker, size) in style.items():
        info = snaps[key]
        ax.scatter([info["node_x"]], [info["node_y"]], s=size, color=color,
                   marker=marker, zorder=5, edgecolors="white")
        if not label.startswith("("):
            ax.annotate(label, (info["node_x"], info["node_y"]),
                        textcoords="offset points", xytext=(10, 8),
                        fontsize=8.5, color="#1c2430")

    ax.set_aspect("equal")
    ax.set_title(
        "Привычный маршрут: трамвай/метро от дома до Октябрьской, затем пешком до МИСИС\n"
        "сплошные = пешком · пунктир = трамвай/метро (не роутим)",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
