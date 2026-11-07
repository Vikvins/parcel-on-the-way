#!/usr/bin/env python3
"""Download large-area walk graph in tiles (Overpass is flaky on big queries)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    LARGE_BBOX_WGS84,
    OUT_DIR,
    RAW_DIR,
    OVERPASS_BASE,
    USER_AGENT,
    add_travel_times,
    ensure_edge_geometry,
    largest_weakly_connected,
)

TILE_CACHE = RAW_DIR / "osm_tiles"
TILE_CACHE.mkdir(parents=True, exist_ok=True)


def main():
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.cache_folder = RAW_DIR / "osm_cache"
    ox.settings.overpass_url = OVERPASS_BASE
    ox.settings.http_user_agent = USER_AGENT
    ox.settings.requests_timeout = 300

    x0, y0, x1, y1 = LARGE_BBOX_WGS84
    tiles = []
    for i, (xa, xb) in enumerate([(x0, (x0 + x1) / 2), ((x0 + x1) / 2, x1)]):
        for j, (ya, yb) in enumerate([(y0, (y0 + y1) / 2), ((y0 + y1) / 2, y1)]):
            tiles.append((f"t{i}{j}", box(xa, ya, xb, yb)))

    graphs = []
    for name, poly in tiles:
        path = TILE_CACHE / f"{name}.graphml"
        if path.exists():
            g = ox.load_graphml(path)
            print(f"tile {name}: cached {g.number_of_nodes()} nodes", flush=True)
            graphs.append(g)
            continue
        t0 = time.time()
        print(f"tile {name} downloading...", flush=True)
        try:
            g = ox.graph_from_polygon(
                poly, network_type="walk", simplify=True,
                retain_all=False, truncate_by_edge=True,
            )
            ox.save_graphml(g, path)
            print(f"  ok {g.number_of_nodes()} nodes in {time.time()-t0:.1f}s", flush=True)
            graphs.append(g)
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {e} after {time.time()-t0:.1f}s", flush=True)

    print(f"\nmerged tiles: {len(graphs)}/{len(tiles)}")
    if len(graphs) < len(tiles):
        print("Not all tiles ready — rerun to retry missing ones.")
        sys.exit(1)

    merged = nx.compose_all(graphs)
    print(f"composed: {merged.number_of_nodes()} nodes, {merged.number_of_edges()} edges")
    g = ox.project_graph(merged)
    ensure_edge_geometry(g)
    add_travel_times(g)
    g = largest_weakly_connected(g)
    print(f"final: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges, CRS {g.graph['crs']}")

    # save in the same format as build_study_graph
    safe = g.copy()
    for _, _, _, data in safe.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] is not None:
            data["geometry"] = data["geometry"].wkt
    ox.save_graphml(safe, OUT_DIR / "moscow_walk.graphml")
    nodes, edges = ox.graph_to_gdfs(g, nodes=True, edges=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    nodes.to_file(OUT_DIR / "moscow_walk.gpkg", layer="nodes", driver="GPKG")
    edges.to_file(OUT_DIR / "moscow_walk.gpkg", layer="edges", driver="GPKG")
    print(f"saved {OUT_DIR / 'moscow_walk.graphml'}")

    # Dijkstra timing
    from parcel_on_the_way.graph import MAX_WALK_MIN, dijkstra_with_prev, snap_to_node, HOME_LON_LAT

    home = snap_to_node(g, HOME_LON_LAT)["node"]
    for label, maxc in (("no-cutoff", None), (f"{MAX_WALK_MIN:.0f}min-cutoff", MAX_WALK_MIN)):
        t0 = time.time()
        d, _ = dijkstra_with_prev(g, home, max_cost=maxc)
        print(f"  Dijkstra {label}: {time.time()-t0:.3f}s, {len(d)} nodes")


if __name__ == "__main__":
    main()
