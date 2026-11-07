#!/usr/bin/env python3
"""Build the Moscow corridor walking graph and report size / timing."""

from __future__ import annotations

import heapq
import math
import sys
import time as time_mod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way import graph as gmod  # noqa: E402
from parcel_on_the_way.graph import (  # noqa: E402
    AKADEMICHESKAYA_LON_LAT,
    MISIS_LON_LAT,
    OKTYABRSKAYA_LON_LAT,
    build_study_graph,
    load_graph,
)


def nearest_node(graph, lon_lat: tuple[float, float]) -> int:
    """Return the nearest graph node to a WGS84 lon/lat pair."""
    import osmnx as ox
    from pyproj import Transformer

    crs = graph.graph["crs"]
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = transformer.transform(lon_lat[0], lon_lat[1])
    return ox.distance.nearest_nodes(graph, X=x, Y=y)


def dijkstra_times(graph, source: int, weight: str = "travel_time_min"):
    """Single-source shortest times to all nodes (plain Dijkstra)."""
    dist = {source: 0.0}
    prev = {}
    queue = [(0.0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d != dist.get(node):
            continue
        for _, nbr, _key, data in graph.out_edges(node, keys=True, data=True):
            cand = d + float(data[weight])
            if cand < dist.get(nbr, math.inf):
                dist[nbr] = cand
                prev[nbr] = node
                heapq.heappush(queue, (cand, nbr))
    return dist, prev


def path_minutes(dist, prev, source: int, target: int) -> float:
    if target not in dist:
        return math.inf
    return dist[target]


def main():
    print("=== 1/2 Build graph ===")
    graph, stats = build_study_graph(save=True)

    print("\n=== 2/2 Smoke-test: two Dijkstra timings (detour primitive) ===")
    # Reload from disk to measure the realistic path used later.
    g = load_graph()
    print(f"Reloaded: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

    # Landmark nodes for the two habitual walking parts:
    #   home → Akademicheskaya, Oktyabrskaya → MISIS
    home_n = nearest_node(g, gmod.HOME_LON_LAT)
    akad_n = nearest_node(g, AKADEMICHESKAYA_LON_LAT)
    okt_n = nearest_node(g, OKTYABRSKAYA_LON_LAT)
    misis_n = nearest_node(g, MISIS_LON_LAT)
    print(f"Nearest nodes (projected graph ids):")
    print(f"  home          → {home_n}")
    print(f"  Akademicheskaya → {akad_n}")
    print(f"  Oktyabrskaya  → {okt_n}")
    print(f"  MISIS         → {misis_n}")

    # Timing: one pair of Dijkstra runs (A→all, B→all on reverse) = one detour batch.
    for label, a, b in (
        ("home → Akademicheskaya", home_n, akad_n),
        ("Oktyabrskaya → MISIS", okt_n, misis_n),
        ("home → MISIS (full corridor)", home_n, misis_n),
    ):
        t0 = time_mod.perf_counter()
        dist_a, _ = dijkstra_times(g, a)
        t1 = time_mod.perf_counter()
        # reverse view (no copy): "to B" via out_edges of the reverse graph
        dist_to_b, _ = dijkstra_times(g.reverse(copy=False), b)
        t2 = time_mod.perf_counter()
        base = dist_a.get(b, math.inf)
        # detour of the endpoint itself must be ~0
        detour_b = dist_a.get(b, math.inf) + dist_to_b.get(b, 0.0) - base
        print(
            f"\n{label}:"
            f"\n  t(A→B) = {base:.2f} min"
            f"\n  Dijkstra from A: {t1 - t0:.3f} s"
            f"\n  Dijkstra to B (reverse): {t2 - t1:.3f} s"
            f"\n  two-run total: {t2 - t0:.3f} s"
            f"\n  detour(B) sanity: {detour_b:.4f} min (expect ~0)"
        )

    # Straight-line distances for orientation
    import osmnx as ox

    xs = {n: g.nodes[n]["x"] for n in (home_n, akad_n, okt_n, misis_n)}
    ys = {n: g.nodes[n]["y"] for n in (home_n, akad_n, okt_n, misis_n)}

    def metric(u, v):
        return math.hypot(xs[u] - xs[v], ys[u] - ys[v])

    print("\nStraight-line (projected metres):")
    print(f"  home–Akademicheskaya: {metric(home_n, akad_n):.0f} m")
    print(f"  Oktyabrskaya–MISIS:   {metric(okt_n, misis_n):.0f} m")
    print(f"  home–MISIS:           {metric(home_n, misis_n):.0f} m")

    print("\nDone. Intermediate files in data/processed/ and data/raw/metadata.json")


if __name__ == "__main__":
    main()
