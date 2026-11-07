#!/usr/bin/env python3
"""Rebuild with +500 m margin, verify landmark snaps, plot Oktyabrskaya→MISIS."""

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

from parcel_on_the_way import graph as gmod  # noqa: E402
from parcel_on_the_way.graph import (  # noqa: E402
    AKADEMICHESKAYA_LON_LAT,
    AKADEMICHESKAYA_STATION_LON_LAT,
    BBOX_MARGIN_M,
    HOME_LON_LAT,
    MISIS_LON_LAT,
    OKTYABRSKAYA_LON_LAT,
    OKTYABRSKAYA_STATION_LON_LAT,
    build_study_graph,
    dijkstra_with_prev,
    load_graph,
    path_coordinates,
    project_lon_lat,
    reconstruct_path,
    snap_to_node,
)

OUT_ZOOM = ROOT / "data" / "processed" / "route_oktyabrskaya_misis.png"


def print_coords_table():
    rows = [
        ("MISIS (university POI, Leninsky 4)", MISIS_LON_LAT,
         "Nominatim: National University of Science and Technology MISIS"),
        ("Home", HOME_LON_LAT,
         "Nominatim: жилой дом (адрес скрыт)"),
        ("Oktyabrskaya street exit ref=2", OKTYABRSKAYA_LON_LAT,
         "OSM node 3583537400 railway=subway_entrance"),
        ("Akademicheskaya street exit ref=6", AKADEMICHESKAYA_LON_LAT,
         "OSM node 282413444 railway=subway_entrance"),
        ("Oktyabrskaya station node", OKTYABRSKAYA_STATION_LON_LAT,
         "OSM node 277499853 railway=station"),
        ("Akademicheskaya station node", AKADEMICHESKAYA_STATION_LON_LAT,
         "OSM node 282413851 railway=station"),
    ]
    print("=== Landmark coordinates (hardcoded after geocoder/OSM lookup) ===")
    for name, (lon, lat), source in rows:
        print(f"  {name}")
        print(f"    lon={lon:.7f}  lat={lat:.7f}")
        print(f"    source: {source}")


def snap_report(g, label, lon_lat):
    info = snap_to_node(g, lon_lat)
    deg = g.degree(info["node"])
    print(
        f"  {label:36s} node={info['node']:<12} "
        f"snap={info['snap_dist_m']:6.1f} m  deg={deg}"
    )
    return info


def route_minutes(g, a, b):
    t0 = time_mod.perf_counter()
    dist, prev = dijkstra_with_prev(g, a)
    t1 = time_mod.perf_counter()
    path = reconstruct_path(prev, a, b)
    t2 = time_mod.perf_counter()
    return dist.get(b, math.inf), path, t1 - t0, t2 - t1


def main():
    print("=== 0) Coordinates and sources ===")
    print_coords_table()

    print("\n=== 1) Rebuild graph with +500 m bbox margin ===")
    print(f"BBOX_MARGIN_M = {BBOX_MARGIN_M}")
    t_build = time_mod.perf_counter()
    g_new, stats = build_study_graph(save=True)
    print(f"(total rebuild wall time includes OSMnx; internal stats above)")

    print("\n=== 2) Snapping check ===")
    g = load_graph()
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, CRS {g.graph['crs']}")
    snaps = {}
    for label, lon_lat in [
        ("MISIS", MISIS_LON_LAT),
        ("Home", HOME_LON_LAT),
        ("Oktyabrskaya exit ref=2", OKTYABRSKAYA_LON_LAT),
        ("Akademicheskaya exit ref=6", AKADEMICHESKAYA_LON_LAT),
        ("Oktyabrskaya station", OKTYABRSKAYA_STATION_LON_LAT),
        ("Akademicheskaya station", AKADEMICHESKAYA_STATION_LON_LAT),
    ]:
        snaps[label] = snap_report(g, label, lon_lat)

    # Connectivity: all landmark nodes must be in the same weak component
    # (we already keep only the largest, so all present nodes are in it).
    print("  (graph is already restricted to the largest weak component)")

    print("\n=== 3) Habitual routes (walking time) ===")
    pairs = [
        ("Home → Akademicheskaya exit", snaps["Home"]["node"], snaps["Akademicheskaya exit ref=6"]["node"]),
        ("Akademicheskaya exit → Home", snaps["Akademicheskaya exit ref=6"]["node"], snaps["Home"]["node"]),
        ("Oktyabrskaya exit → MISIS", snaps["Oktyabrskaya exit ref=2"]["node"], snaps["MISIS"]["node"]),
        ("MISIS → Oktyabrskaya exit", snaps["MISIS"]["node"], snaps["Oktyabrskaya exit ref=2"]["node"]),
        ("Home → MISIS (full corridor)", snaps["Home"]["node"], snaps["MISIS"]["node"]),
    ]
    print(f"  {'route':34s} {'t_min':>8s} {'straight_m':>11s} {'Dijkstra_s':>11s}")
    route_geoms = {}
    for label, a, b in pairs:
        tmin, path, dt, _ = route_minutes(g, a, b)
        ax, ay = g.nodes[a]["x"], g.nodes[a]["y"]
        bx, by = g.nodes[b]["x"], g.nodes[b]["y"]
        straight = math.hypot(bx - ax, by - ay)
        print(f"  {label:34s} {tmin:8.2f} {straight:11.0f} {dt:11.3f}")
        route_geoms[label] = path

    print("\n=== 4) Zoomed map: Oktyabrskaya exit → MISIS ===")
    path = route_geoms["Oktyabrskaya exit → MISIS"]
    coords = path_coordinates(g, path)
    tmin = route_minutes(g, snaps["Oktyabrskaya exit ref=2"]["node"], snaps["MISIS"]["node"])[0]

    fig, ax = plt.subplots(figsize=(9, 9))
    # background edges near the route bbox
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    pad = 180.0
    x0, x1 = min(xs) - pad, max(xs) + pad
    y0, y1 = min(ys) - pad, max(ys) + pad
    for u, v, _k, data in g.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        c = list(geom.coords)
        if max(p[0] for p in c) < x0 or min(p[0] for p in c) > x1:
            continue
        if max(p[1] for p in c) < y0 or min(p[1] for p in c) > y1:
            continue
        ax.plot([p[0] for p in c], [p[1] for p in c], color="#c5d4de",
                linewidth=0.6, zorder=1)

    ax.plot(xs, ys, color="#c45c26", linewidth=3.2, zorder=3,
            label=f"маршрут, {tmin:.1f} мин")

    # landmarks
    for label, lon_lat, color, marker in [
        ("Октябрьская (выход)", OKTYABRSKAYA_LON_LAT, "#1f6f8b", "o"),
        ("МИСИС", MISIS_LON_LAT, "#c45c26", "s"),
    ]:
        x, y = project_lon_lat(g, lon_lat)
        snap = snaps["Oktyabrskaya exit ref=2" if "Октябр" in label else "MISIS"]
        ax.scatter([snap["node_x"]], [snap["node_y"]], s=90, color=color,
                   marker=marker, zorder=5, edgecolors="white")
        ax.annotate(
            f"{label}\nsnap {snap['snap_dist_m']:.0f} m",
            (snap["node_x"], snap["node_y"]),
            textcoords="offset points", xytext=(10, 8), fontsize=9,
            color="#1c2430",
        )

    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_title(
        f"Октябрьская (уличный выход) → МИСИС\n"
        f"время {tmin:.2f} мин · {len(path) - 1} рёбер · скорость 5 км/ч",
        fontsize=11,
    )
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_ZOOM, dpi=160)
    print(f"Saved {OUT_ZOOM}")

    print("\n=== Suspicious checks ===")
    okt_to_misis = tmin
    if okt_to_misis > 8.0:
        print(f"  WARNING: Oktyabrskaya→MISIS is {okt_to_misis:.1f} min (expected 5–8).")
    else:
        print(f"  OK: Oktyabrskaya→MISIS = {okt_to_misis:.1f} min (expected 5–8).")
    for label, info in snaps.items():
        if info["snap_dist_m"] > 40:
            print(f"  WARNING: {label} snap distance {info['snap_dist_m']:.1f} m is large.")
        deg = g.degree(info["node"])
        if deg <= 1:
            print(f"  WARNING: {label} snaps to a degree-{deg} node (possible stub/courtyard).")


if __name__ == "__main__":
    main()
