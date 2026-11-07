"""Walking graph construction and travel-time weights.

Reuses the OSMnx walk-graph pipeline idea from VikWay, but targets a Moscow
corridor and attaches a constant walking speed so every edge has a time cost.
"""

from __future__ import annotations

import json
import math
import time as time_mod
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx
import osmnx as ox
from shapely.geometry import LineString, box

# --- Study corridor ---------------------------------------------------------
# NUST MISIS (Leninsky Prospekt 4) → home (residential building, address withheld).
#
# Coordinate sources (documented — all looked up 2026-10-15, NOT hand-guessed):
#   MISIS university POI   — Nominatim: "National University of Science and
#                            Technology MISIS" → amenity=university on
#                            Leningradsky/Leninsky Prospekt, 55.7279401, 37.6077966
#   Oktyabrskaya street    — OSM node 3583537400, railway=subway_entrance ref=2
#     exit (toward MISIS)    55.7294105, 37.6108935 (street-level exit)
#   Akademicheskaya street — OSM node 282413444, railway=subway_entrance ref=6
#     exit (north, toward home) 55.6888255, 37.5741449 (street-level exit)
#   Akademicheskaya station— OSM node 282413851, 55.6878077, 37.5735148
#   Home building          — residential building (address withheld) → 55.6980013, 37.5791629

MISIS_LON_LAT = (37.6077966, 55.7279401)  # university POI / Leninsky Prospekt 4
HOME_LON_LAT = (37.5791629, 55.6980013)  # residential building (address withheld)

# User-confirmed habitual access points (names as the user says them):
#   tram near home: "проспект 60-летия Октября"
#   metro from home: "Ленинский проспект (южный вход)"
# OSM records these as:
#   tram platform / stop_position «Проспект 60-летия Октября»
#   subway_entrance ref=1 at Leninsky Prospekt station (south)
TRAM_60LETIYA_LON_LAT = (37.5801918, 55.7023082)  # OSM 1376583452 stop_position
METRO_LENINSKY_SOUTH_LON_LAT = (37.5851442, 55.7068542)  # OSM 3198614338 entrance ref=1

# Tram stop near MISIS is NOT part of the commute: the user walks from
# Oktyabrskaya metro exit to MISIS. Tram is only home → Leninsky Prospekt.
# Still kept on the map as an OSM reference point.
TRAM_OKTYABRSKAYA_LON_LAT = (37.6116715, 55.7268934)
# Street exit of Oktyabrskaya metro toward MISIS:
OKTYABRSKAYA_LON_LAT = (37.6108935, 55.7294105)  # OSM 3583537400 ref=2

# Kept for reference / older scripts:
AKADEMICHESKAYA_LON_LAT = (37.5741449, 55.6888255)  # OSM 282413444 ref=6
OKTYABRSKAYA_STATION_LON_LAT = (37.6109093, 55.7304961)
AKADEMICHESKAYA_STATION_LON_LAT = (37.5735148, 55.6878077)
TRAM_UDARNITSA_LON_LAT = (37.6106726, 55.7229183)  # alt stop near MISIS, for the map

# Named landmarks for reports / maps.
LANDMARKS = {
    "home": HOME_LON_LAT,
    "misis": MISIS_LON_LAT,
    "tram_60letiya": TRAM_60LETIYA_LON_LAT,  # «Проспект 60-летия Октября»
    "metro_leninsky_south": METRO_LENINSKY_SOUTH_LON_LAT,  # «Ленинский проспект (южный вход)»
    "tram_oktyabrskaya": TRAM_OKTYABRSKAYA_LON_LAT,  # tram near MISIS
    "oktyabrskaya_exit": OKTYABRSKAYA_LON_LAT,
    "tram_udarnitsa": TRAM_UDARNITSA_LON_LAT,
}

LANDMARK_SOURCES = {
    "home": "жилой дом рядом с пр-том 60-летия Октября (адрес скрыт)",
    "misis": "Nominatim: National University of Science and Technology MISIS",
    "tram_60letiya": "OSM node 1376583452 railway=tram_stop «Проспект 60-летия Октября»",
    "metro_leninsky_south": "OSM node 3198614338 railway=subway_entrance ref=1 (south)",
    "tram_oktyabrskaya": "OSM node 1376510974 railway=tram_stop «Метро «Октябрьская»»",
    "oktyabrskaya_exit": "OSM node 3583537400 railway=subway_entrance ref=2",
    "tram_udarnitsa": "OSM node 1376510980 railway=tram_stop «Фабрика «Ударница»»",
}

# Walking legs of the habitual commute (tram/metro rides are NOT in the graph).
#
# Two commute options (user confirmed):
#   Option 1 (tram + metro):  home → [tram] «60-летия Октября» → «Ленинский
#       проспект (южный)» → [metro] → «Октябрьская» → MISIS
#   Option 2 (metro):         home → [walk] «Ленинский проспект (южный)» →
#       [metro] → «Октябрьская» → MISIS
# Tram is ONLY home → Leninsky Prospekt. The last leg is always walking from
# Oktyabrskaya metro exit to MISIS.
#
# Scenarios pick a pickup point across ALL walking legs of ALL options.
HABITUAL_WALK_LEGS = (
    ("home → tram «Проспект 60-летия Октября»", "home", "tram_60letiya"),
    ("home → metro «Ленинский проспект (южный)»", "home", "metro_leninsky_south"),
    ("metro «Октябрьская» exit → MISIS", "oktyabrskaya_exit", "misis"),
)

# Full commute itineraries as ordered segments (for scenario explanations).
# transport = "walk" is on the graph; "tram"/"metro" is a transfer (not routed).
HABITUAL_COMMUTES = (
    {
        "id": "tram_metro",
        "label": "трамвай + метро",
        "segments": (
            {"mode": "walk", "from": "home", "to": "tram_60letiya",
             "label": "пешком до трамвая «60-летия Октября»"},
            {"mode": "tram", "from": "tram_60letiya", "to": "metro_leninsky_south",
             "label": "трамвай до м. «Ленинский проспект»"},
            {"mode": "metro", "from": "metro_leninsky_south", "to": "oktyabrskaya_exit",
             "label": "метро до м. «Октябрьская»"},
            {"mode": "walk", "from": "oktyabrskaya_exit", "to": "misis",
             "label": "пешком от м. «Октябрьская» до МИСИС"},
        ),
    },
    {
        "id": "metro_only",
        "label": "только метро",
        "segments": (
            {"mode": "walk", "from": "home", "to": "metro_leninsky_south",
             "label": "пешком до м. «Ленинский проспект (южный)»"},
            {"mode": "metro", "from": "metro_leninsky_south", "to": "oktyabrskaya_exit",
             "label": "метро до м. «Октябрьская»"},
            {"mode": "walk", "from": "oktyabrskaya_exit", "to": "misis",
             "label": "пешком от м. «Октябрьская» до МИСИС"},
        ),
    },
)

# Rectangle along the corridor: ~5 km long, ~2 km wide.
CORRIDOR_HALF_LENGTH_M = 2600.0
CORRIDOR_HALF_WIDTH_M = 1100.0
# Extra margin on every side of the axis-aligned bbox (requested +500 m).
BBOX_MARGIN_M = 500.0

# Dijkstra time cutoff: points farther than this on foot are "not on the way".
MAX_WALK_MIN = 30.0

# Expanded area (several districts): ~8×11 km covering Yakimanka, Khamovniki,
# Donskoy, Gagarinsky, Akademichesky. Used when build_study_graph(large=True).
LARGE_BBOX_WGS84 = (37.530, 55.665, 37.660, 55.755)

WALK_SPEED_KMH = 5.0
WALK_SPEED_M_PER_MIN = WALK_SPEED_KMH * 1000.0 / 60.0  # 83.333 m/min

CRS_OSM = "EPSG:4326"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "data" / "processed"
TIMEOUT_S = 180


def study_area_box(large: bool = False) -> tuple[tuple[float, float, float, float], object]:
    """Return (minx, miny, maxx, maxy) in WGS84 and a projected-ready polygon.
    If large=True, use the multi-district box instead of the corridor rectangle.

    The rectangle is aligned with the MISIS→home direction in a local metric
    approximation (1 deg lat ≈ 111320 m, 1 deg lon at 55.7° ≈ 62200 m).
    """
    if large:
        bounds = LARGE_BBOX_WGS84
        return bounds, box(*bounds)

    lon1, lat1 = MISIS_LON_LAT
    lon2, lat2 = HOME_LON_LAT
    mid_lon = (lon1 + lon2) / 2.0
    mid_lat = (lat1 + lat2) / 2.0

    # Direction from home to MISIS (unit vector in metric space).
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))
    dx = (lon1 - lon2) * m_per_deg_lon
    dy = (lat1 - lat2) * m_per_deg_lat
    length = math.hypot(dx, dy)
    if length < 1.0:
        raise RuntimeError("MISIS and home coordinates are too close")
    ux, uy = dx / length, dy / length
    # Perpendicular in metric space.
    px, py = -uy, ux

    # Four corners of the oriented rectangle around the midpoint.
    half_l = CORRIDOR_HALF_LENGTH_M
    half_w = CORRIDOR_HALF_WIDTH_M
    corners_m = []
    for along, across in (
        (-half_l, -half_w),
        (-half_l, +half_w),
        (+half_l, +half_w),
        (+half_l, -half_w),
    ):
        cx = mid_lon * m_per_deg_lon + along * ux + across * px
        cy = mid_lat * m_per_deg_lat + along * uy + across * py
        corners_m.append((cx / m_per_deg_lon, cy / m_per_deg_lat))

    lons = [c[0] for c in corners_m]
    lats = [c[1] for c in corners_m]
    # Expand the axis-aligned bbox by BBOX_MARGIN_M on every side.
    margin_lat = BBOX_MARGIN_M / m_per_deg_lat
    margin_lon = BBOX_MARGIN_M / m_per_deg_lon
    bounds = (
        min(lons) - margin_lon,
        min(lats) - margin_lat,
        max(lons) + margin_lon,
        max(lats) + margin_lat,
    )
    polygon = box(*bounds)
    return bounds, polygon


# overpass-api.de fails with SSL EOF on this machine; use a public mirror.
# OSMnx appends "/interpreter" to overpass_url, so store the base only.
OVERPASS_BASE = "https://maps.mail.ru/osm/tools/overpass/api"
USER_AGENT = "parcel-on-the-way-prototype/0.1 (student research)"


def download_walk_graph(polygon_wgs84):
    """Download a simplified walk graph clipped to the study polygon."""
    ox.settings.use_cache = True
    ox.settings.log_console = False
    ox.settings.cache_folder = RAW_DIR / "osm_cache"
    ox.settings.overpass_url = OVERPASS_BASE
    ox.settings.http_user_agent = USER_AGENT
    ox.settings.requests_timeout = TIMEOUT_S
    graph = ox.graph_from_polygon(
        polygon_wgs84,
        network_type="walk",
        simplify=True,
        retain_all=False,
        truncate_by_edge=True,
    )
    return ox.project_graph(graph)  # auto UTM for Moscow


def ensure_edge_geometry(graph):
    """Give every edge a geometry and a metric length in metres."""
    for u, v, key, data in graph.edges(keys=True, data=True):
        if data.get("geometry") is None:
            data["geometry"] = LineString(
                [
                    (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                    (graph.nodes[v]["x"], graph.nodes[v]["y"]),
                ]
            )
        data["length"] = float(data["geometry"].length)


def add_travel_times(graph, speed_m_per_min: float = WALK_SPEED_M_PER_MIN):
    """Attach travel_time_min = length_m / walking_speed to every edge."""
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["travel_time_min"] = float(data["length"]) / speed_m_per_min


def largest_weakly_connected(graph):
    """Keep only the largest weakly connected component."""
    component = max(nx.weakly_connected_components(graph), key=len)
    return graph.subgraph(component).copy()


def write_metadata(graph, bounds, polygon, elapsed_s: float):
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "study_area": {
            "misis_lon_lat": MISIS_LON_LAT,
            "home_lon_lat": HOME_LON_LAT,
            "oktyabrskaya_exit_lon_lat": OKTYABRSKAYA_LON_LAT,
            "akademicheskaya_exit_lon_lat": AKADEMICHESKAYA_LON_LAT,
            "oktyabrskaya_station_lon_lat": OKTYABRSKAYA_STATION_LON_LAT,
            "akademicheskaya_station_lon_lat": AKADEMICHESKAYA_STATION_LON_LAT,
            "bbox_wgs84": list(bounds),
            "corridor_half_length_m": CORRIDOR_HALF_LENGTH_M,
            "corridor_half_width_m": CORRIDOR_HALF_WIDTH_M,
            "bbox_margin_m": BBOX_MARGIN_M,
        },
        "coordinate_sources": {
            "misis": "Nominatim: National University of Science and Technology MISIS",
            "home": "жилой дом рядом с пр-том 60-летия Октября (адрес скрыт)",
            "oktyabrskaya_exit": "OSM node 3583537400 railway=subway_entrance ref=2",
            "akademicheskaya_exit": "OSM node 282413444 railway=subway_entrance ref=6",
        },
        "walk_speed_kmh": WALK_SPEED_KMH,
        "walk_speed_m_per_min": WALK_SPEED_M_PER_MIN,
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges_directed": graph.number_of_edges(),
        "build_elapsed_s": round(elapsed_s, 2),
        "crs": str(graph.graph.get("crs", "unknown")),
        "osm_retrieval": "live Overpass endpoint used by OSMnx",
        "attribution": "© OpenStreetMap contributors, ODbL 1.0",
        "note": "Prototype area is fictional development data for pickup points; "
        "do not treat sample CSV as real CDEK locations.",
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_graph(graph):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # GraphML cannot store Shapely geometries directly.
    safe = graph.copy()
    for _, _, _, data in safe.edges(keys=True, data=True):
        if "geometry" in data and data["geometry"] is not None:
            data["geometry"] = data["geometry"].wkt
    ox.save_graphml(safe, OUT_DIR / "moscow_walk.graphml")
    nodes, edges = ox.graph_to_gdfs(graph, nodes=True, edges=True)
    nodes.to_file(OUT_DIR / "moscow_walk.gpkg", layer="nodes", driver="GPKG")
    edges.to_file(OUT_DIR / "moscow_walk.gpkg", layer="edges", driver="GPKG")


def build_study_graph(save: bool = True, large: bool = False) -> tuple[nx.MultiDiGraph, dict]:
    """End-to-end build. Returns (graph, stats)."""
    t0 = time_mod.perf_counter()
    bounds, polygon = study_area_box(large=large)
    print(
        f"Study bbox WGS84: "
        f"lon[{bounds[0]:.5f}, {bounds[2]:.5f}] "
        f"lat[{bounds[1]:.5f}, {bounds[3]:.5f}]"
    )
    print(f"Corridor: {2 * CORRIDOR_HALF_LENGTH_M:.0f} m × {2 * CORRIDOR_HALF_WIDTH_M:.0f} m")

    graph = download_walk_graph(polygon)
    ensure_edge_geometry(graph)
    add_travel_times(graph)
    graph = largest_weakly_connected(graph)
    elapsed = time_mod.perf_counter() - t0

    nodes = graph.number_of_nodes()
    edges = graph.number_of_edges()
    lengths = [d["length"] for _, _, _, d in graph.edges(keys=True, data=True)]
    times = [d["travel_time_min"] for _, _, _, d in graph.edges(keys=True, data=True)]
    stats = {
        "nodes": nodes,
        "edges_directed": edges,
        "build_elapsed_s": elapsed,
        "crs": str(graph.graph.get("crs")),
        "median_edge_m": sorted(lengths)[len(lengths) // 2],
        "median_edge_min": sorted(times)[len(times) // 2],
        "walk_speed_kmh": WALK_SPEED_KMH,
    }
    print(
        f"Graph: {nodes} nodes, {edges} directed edges, "
        f"CRS {stats['crs']}"
    )
    print(
        f"Median edge: {stats['median_edge_m']:.1f} m "
        f"≈ {stats['median_edge_min']:.2f} min at {WALK_SPEED_KMH} km/h"
    )
    print(f"Build time: {elapsed:.2f} s")

    if save:
        save_graph(graph)
        write_metadata(graph, bounds, polygon, elapsed)
        print(f"Saved: {OUT_DIR / 'moscow_walk.graphml'}")
        print(f"Saved: {OUT_DIR / 'moscow_walk.gpkg'}")
        print(f"Saved: {RAW_DIR / 'metadata.json'}")
    return graph, stats


def load_graph() -> nx.MultiDiGraph:
    """Load the processed GraphML and restore geometry / float weights."""
    path = OUT_DIR / "moscow_walk.graphml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/01_build_graph.py first."
        )
    graph = ox.load_graphml(path)
    from shapely import wkt

    for _, _, _, data in graph.edges(keys=True, data=True):
        geom = data.get("geometry")
        if isinstance(geom, str):
            data["geometry"] = wkt.loads(geom)
        data["length"] = float(data["length"])
        data["travel_time_min"] = float(data["travel_time_min"])
    return graph


def project_lon_lat(graph, lon_lat: tuple[float, float]) -> tuple[float, float]:
    """Transform WGS84 lon/lat into the graph CRS."""
    from pyproj import Transformer

    transformer = Transformer.from_crs(CRS_OSM, graph.graph["crs"], always_xy=True)
    x, y = transformer.transform(lon_lat[0], lon_lat[1])
    return float(x), float(y)


def snap_to_node(graph, lon_lat: tuple[float, float]) -> dict:
    """Snap a WGS84 point to the nearest graph node and report distance."""
    x, y = project_lon_lat(graph, lon_lat)
    node = ox.distance.nearest_nodes(graph, X=x, Y=y)
    node = int(node)
    nx_ = float(graph.nodes[node]["x"])
    ny_ = float(graph.nodes[node]["y"])
    dist_m = float(math.hypot(nx_ - x, ny_ - y))
    return {
        "node": node,
        "snap_dist_m": dist_m,
        "node_x": nx_,
        "node_y": ny_,
        "lon_lat": lon_lat,
    }


def dijkstra_with_prev(
    graph,
    source: int,
    weight: str = "travel_time_min",
    max_cost: float | None = None,
):
    """Single-source shortest paths; returns (dist, prev).

    If max_cost is set (e.g. 30 min walking), nodes farther than that are not
    expanded — points beyond the cutoff are "not on the way".
    """
    import heapq

    dist = {source: 0.0}
    prev: dict[int, int] = {}
    queue = [(0.0, source)]
    while queue:
        d, node = heapq.heappop(queue)
        if d != dist.get(node):
            continue
        if max_cost is not None and d >= max_cost:
            continue
        for _, nbr, _key, data in graph.out_edges(node, keys=True, data=True):
            cand = d + float(data[weight])
            if max_cost is not None and cand > max_cost:
                continue
            if cand < dist.get(nbr, math.inf):
                dist[nbr] = cand
                prev[nbr] = node
                heapq.heappush(queue, (cand, nbr))
    return dist, prev


def reconstruct_path(prev: dict[int, int], source: int, target: int) -> list[int]:
    """Node sequence source→target from a Dijkstra prev map."""
    if target == source:
        return [source]
    if target not in prev and target != source:
        raise nx.NetworkXNoPath(f"No path from {source} to {target}")
    path = [target]
    node = target
    while node != source:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path


def path_coordinates(graph, node_path: list[int]) -> list[tuple[float, float]]:
    """Concatenate edge geometries along a node path (projected metres)."""
    coords: list[tuple[float, float]] = []
    for u, v in zip(node_path[:-1], node_path[1:]):
        # pick any parallel edge that has geometry
        edge_data = None
        for _k, data in graph.get_edge_data(u, v).items():
            edge_data = data
            break
        geom = edge_data.get("geometry")
        if geom is not None:
            points = list(geom.coords)
        else:
            points = [
                (graph.nodes[u]["x"], graph.nodes[u]["y"]),
                (graph.nodes[v]["x"], graph.nodes[v]["y"]),
            ]
        if coords and coords[-1] == points[0]:
            coords.extend(points[1:])
        else:
            coords.extend(points)
    return coords
