#!/usr/bin/env python3
"""Export the interactive demo page (no build step, no server-side code).

  PYTHONPATH=src .venv/bin/python scripts/09_export_demo.py --csv data/results/points_used.csv

Writes two files next to the vendored Leaflet:
  demo/index.html  the page (template below)
  demo/data.js     walking graph moscow_walk.graphml packed into typed arrays,
                   CDEK points with hours, metro exits and named places

The page lets the user pick start and finish on the map and recomputes the
detour of every point in the browser with the same formula as detour.py:
detour = t(A→P) + t(P→B) − t(A→B), Dijkstra on travel_time_min (5 km/h),
4 min handover, no waiting before opening (hours.is_open_at). Rerun after
updating the points or the graph.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    LANDMARKS,
    MAX_WALK_MIN,
    dijkstra_with_prev,
    load_graph,
    snap_to_node,
)
from parcel_on_the_way.points import WEEKDAYS, load_any_points  # noqa: E402
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402

# Orange line (6) stations inside the study area, north to south.
METRO_LINE = ["Октябрьская", "Шаболовская", "Ленинский проспект", "Академическая"]

PLACES = [  # quick picks for «Откуда / Куда»: id, name, «до …», «по дороге …», hint
    ("home", "Дом", "дома", "домой", "жилой дом у пр-та 60-летия Октября"),
    ("misis", "МИСИС", "МИСИС", "в МИСИС", "Ленинский пр-т, 4"),
    ("oktyabrskaya_exit", "м. Октябрьская", "м. Октябрьская", "к м. Октябрьская", "выход 2"),
    ("metro_leninsky_south", "м. Ленинский проспект", "м. Ленинский проспект",
     "к м. Ленинский проспект", "южный вестибюль"),
    ("shabolovskaya", "м. Шаболовская", "м. Шаболовская", "к м. Шаболовская", "вестибюль"),
    ("novokuznetskaya", "м. Новокузнецкая", "м. Новокузнецкая", "к м. Новокузнецкая", "вестибюль"),
]
# Places that are not model landmarks (approximate station coordinates, lon/lat).
EXTRA_PLACES = {
    "shabolovskaya": (37.60785, 55.71871),
    "novokuznetskaya": (37.62931, 55.74242),
}

# Route presets for the demo. The two examples are made-up trips (only the route
# is invented; points, hours and detours are computed as usual). The example
# endpoints were picked by scanning walking trips from МИСИС for ones with
# several CDEK points within a 5 min detour on a Wednesday evening.
ROUTE_PRESETS = [
    {"id": "ex_novokuz", "label": "Пример: МИСИС → м. Новокузнецкая",
     "sub": "пешком через Замоскворечье", "a": "misis", "b": "novokuznetskaya", "mode": "walk"},
    {"id": "ex_shabol", "label": "Пример: МИСИС → м. Шаболовская",
     "sub": "короткая прогулка", "a": "misis", "b": "shabolovskaya", "mode": "walk"},
    {"id": "vika", "label": "Привычный маршрут: МИСИС → дом",
     "sub": "метро и пешком, как в модели", "a": "misis", "b": "home", "mode": "metro"},
]


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _display_name(name: str) -> str:
    """Address without the brand prefix; bare coordinates become a short label."""
    n = name.strip()
    for prefix in ("СДЭК, ", "СДЭК "):
        if n.startswith(prefix):
            n = n[len(prefix):]
    if n and n.replace(".", "").replace(",", "").replace(" ", "").isdigit():
        return "пункт СДЭК (адрес не указан)"
    return n or "пункт СДЭК"


def _source_ru(points) -> str:
    n = len(points)
    unknown = sum("unknown" in (p.note or "") for p in points)
    site = sum("cdek_site" in (p.note or "") for p in points)
    if site:
        src = f"{n} пунктов СДЭК с сайта cdek.ru, часы работы и лимиты веса на 16.10.2026"
    elif any("osm" in (p.note or "").lower() for p in points):
        src = f"{n} пунктов СДЭК из OpenStreetMap"
    else:
        src = f"{n} пунктов выдачи"
    if unknown:
        src += f"; у {unknown} часы неизвестны и приняты как 10:00–20:00"
    return src


def pack_graph(g) -> dict:
    """CSR adjacency + per-edge geometry, coordinates in 1e-6 degrees."""
    from pyproj import Transformer
    from shapely.geometry import LineString

    tr = Transformer.from_crs(g.graph["crs"], "EPSG:4326", always_xy=True)
    ids = list(g.nodes)
    index = {n: i for i, n in enumerate(ids)}
    xs = np.array([g.nodes[n]["x"] for n in ids])
    ys = np.array([g.nodes[n]["y"] for n in ids])
    lon, lat = tr.transform(xs, ys)
    node_ll = np.empty(2 * len(ids), dtype=np.int32)
    node_ll[0::2] = np.round(np.asarray(lat) * 1e6)
    node_ll[1::2] = np.round(np.asarray(lon) * 1e6)

    # cheapest parallel edge per directed pair
    best: dict[tuple[int, int], tuple[float, object]] = {}
    for u, v, data in g.edges(data=True):
        key = (index[u], index[v])
        w = float(data["travel_time_min"])
        if key not in best or w < best[key][0]:
            best[key] = (w, data.get("geometry"))

    # one geometry per undirected edge, stored from lower to higher node index
    und: dict[tuple[int, int], int] = {}
    geo_off = [0]
    geo_pts: list[int] = []
    for (a, b), (_w, geom) in sorted(best.items()):
        lo, hi = min(a, b), max(a, b)
        if (lo, hi) in und:
            continue
        und[(lo, hi)] = len(und)
        inner: list[tuple[float, float]] = []
        if geom is not None and len(geom.coords) > 2:
            line = LineString(geom.coords).simplify(1.5)
            coords = list(line.coords)
            # orient lo → hi
            sx, sy = g.nodes[ids[lo]]["x"], g.nodes[ids[lo]]["y"]
            if math.hypot(coords[0][0] - sx, coords[0][1] - sy) > math.hypot(
                coords[-1][0] - sx, coords[-1][1] - sy
            ):
                coords.reverse()
            inner = coords[1:-1]
        if inner:
            ilon, ilat = tr.transform([c[0] for c in inner], [c[1] for c in inner])
            for la, lo_ in zip(np.atleast_1d(ilat), np.atleast_1d(ilon)):
                geo_pts += [round(la * 1e6), round(lo_ * 1e6)]
        geo_off.append(len(geo_pts) // 2)

    n = len(ids)
    out_edges: list[list[tuple[int, float, int]]] = [[] for _ in range(n)]
    for (a, b), (w, _g) in best.items():
        out_edges[a].append((b, w, und[(min(a, b), max(a, b))]))
    off = np.zeros(n + 1, dtype=np.uint32)
    to, wt, eid = [], [], []
    for i, lst in enumerate(out_edges):
        for b, w, e in lst:
            to.append(b)
            wt.append(w)
            eid.append(e)
        off[i + 1] = len(to)

    return {
        "n": n,
        "ll": _b64(node_ll),
        "off": _b64(off),
        "to": _b64(np.array(to, dtype=np.uint32)),
        "w": _b64(np.array(wt, dtype=np.float32)),
        "eid": _b64(np.array(eid, dtype=np.uint32)),
        "geoOff": _b64(np.array(geo_off, dtype=np.uint32)),
        "geo": _b64(np.array(geo_pts, dtype=np.int32)),
    }, index, (np.asarray(lat), np.asarray(lon))


def area_outline(lat: np.ndarray, lon: np.ndarray) -> list[list[float]]:
    from shapely.geometry import MultiPoint

    hull = MultiPoint(list(zip(lon, lat))).convex_hull.simplify(0.0005)
    return [[round(y, 6), round(x, 6)] for x, y in hull.exterior.coords]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/results/points_used.csv")
    ap.add_argument("--out", default="demo/index.html")
    args = ap.parse_args()

    cfg = load_habitual_config()
    g = load_graph()
    graph_pack, index, (lat, lon) = pack_graph(g)

    def node_of(lon_lat):
        return index[snap_to_node(g, lon_lat)["node"]]

    points = load_any_points(ROOT / args.csv)
    pts_json = []
    for p in points:
        hours = {}
        for d in WEEKDAYS:
            s, e = p.hours_on(d)
            hours[d] = [s, e] if s and e else None
        pts_json.append({
            "id": p.id,
            "name": _display_name(p.name),
            "lat": round(p.lat, 6),
            "lon": round(p.lon, 6),
            "node": node_of((p.lon, p.lat)),
            "type": p.point_type,
            "maxKg": p.max_weight_kg,
            "hours": hours,
            "hoursGuessed": "unknown" in (p.note or ""),
        })

    metro = []
    import csv

    with open(ROOT / "data" / "transit_stops.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["kind"] != "metro_exit" or r["name"] not in METRO_LINE:
                continue
            ll = (float(r["lon"]), float(r["lat"]))
            metro.append({
                "station": r["name"],
                "exit": r["ref"],
                "lat": ll[1],
                "lon": ll[0],
                "node": node_of(ll),
            })

    places = []
    for key, name, gen, way, hint in PLACES:
        lo, la = EXTRA_PLACES.get(key) or LANDMARKS[key]
        places.append({"id": key, "name": name, "gen": gen, "way": way, "hint": hint,
                       "lat": la, "lon": lo, "node": node_of((lo, la))})

    payload = {
        "graph": graph_pack,
        "points": pts_json,
        "metro": metro,
        "metroLine": METRO_LINE,
        "places": places,
        "presets": ROUTE_PRESETS,
        "area": area_outline(lat, lon),
        "serviceMin": cfg.service_time_min,
        "maxWalkMin": MAX_WALK_MIN,
        "source": _source_ru(points),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    data_js = out.parent / "data.js"
    data_js.write_text("window.POTW = " + json.dumps(payload, ensure_ascii=False) + ";\n",
                       encoding="utf-8")
    out.write_text(TEMPLATE, encoding="utf-8")

    # reference value for checking the in-browser Dijkstra against Python
    home = snap_to_node(g, LANDMARKS["home"])["node"]
    misis = snap_to_node(g, LANDMARKS["misis"])["node"]
    dist, _ = dijkstra_with_prev(g, home)
    print(f"wrote {out} and {data_js.name} ({data_js.stat().st_size / 1e6:.1f} MB)")
    print(f"graph: {graph_pack['n']} nodes; points: {len(pts_json)}; metro exits: {len(metro)}")
    print(f"check: walk home → МИСИС = {dist[misis]:.2f} min")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Посылка по пути</title>
<link rel="stylesheet" href="leaflet/leaflet.css"/>
<style>
  :root {
    --ink: #15181c;
    --text: #3d434b;
    --muted: #6b727c;
    --faint: #a3a9b1;
    --line: #e6e8eb;
    --soft: #f3f4f6;
    --green: #1ab248;
    --green-ink: #0a7a2f;
    --green-strong: #13923c;
    --red: #e5483b;
    --amber-ink: #8a5a00;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body { background: #eef0f2; color: var(--ink); overflow: hidden;
    font: 16px/1.4 -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
    font-variant-numeric: tabular-nums; }
  button, input, select { font: inherit; color: inherit; }
  button { cursor: pointer; }
  #map { position: absolute; inset: 0; }
  [hidden] { display: none !important; }

  /* floating panel, like a maps app */
  .panel { position: absolute; z-index: 1000; top: 16px; left: 16px; width: 400px; max-height: calc(100% - 32px);
           overflow-y: auto; background: #fff; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,.14); }
  .search { display: flex; align-items: center; gap: 12px; width: 100%; padding: 16px 18px; border: 0; background: none;
            text-align: left; border-bottom: 1px solid var(--line); }
  .search span { flex: 1; color: var(--muted); font-size: 17px; }
  .search:hover span { color: var(--ink); }
  .intro { padding: 18px 20px 6px; }
  .intro h1 { margin: 0; font-size: 22px; font-weight: 750; letter-spacing: -0.02em; }
  .intro p { margin: 6px 0 0; color: var(--text); }
  .sh { padding: 16px 20px 6px; font-size: 14px; font-weight: 600; color: var(--muted); }
  .sh.sep { margin-top: 10px; border-top: 1px solid var(--line); padding-top: 18px; }
  .demo-inline { display: none; }
  .demo { position: absolute; z-index: 1000; top: 16px; left: 432px; width: 300px; background: #fff;
          border: 1px solid var(--line); border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,.10); padding: 6px 0 8px; }
  .demo[hidden] { display: none; }
  .demo .dh { padding: 12px 18px 8px; }
  .demo .dh b { display: block; font-size: 15px; font-weight: 650; }
  .demo .dh small { display: block; color: var(--muted); font-size: 13px; margin-top: 2px; }
  .demo .item { padding: 9px 18px; gap: 12px; }
  .demo .item .ic { width: 30px; height: 30px; }
  .demo .item b { font-size: 14px; }
  .demo .item small { font-size: 13px; }
  .item { display: flex; align-items: center; gap: 14px; width: 100%; padding: 11px 20px; border: 0; background: none; text-align: left; }
  .item:hover { background: var(--soft); }
  .item .ic { flex: none; width: 34px; height: 34px; border-radius: 50%; background: var(--soft); display: grid; place-items: center; color: var(--text); }
  .item b { display: block; font-weight: 600; }
  .item small { display: block; color: var(--muted); font-size: 14px; }
  .pad { height: 10px; }

  /* route form */
  .head { display: flex; align-items: center; gap: 6px; padding: 12px 12px 4px; }
  .back { width: 36px; height: 36px; border: 0; background: none; border-radius: 50%; display: grid; place-items: center; }
  .back:hover { background: var(--soft); }
  .head span { font-weight: 700; font-size: 17px; }
  .fields { position: relative; padding: 6px 60px 0 20px; display: grid; gap: 8px; }
  .f { display: flex; align-items: center; gap: 12px; }
  .f i { flex: none; width: 10px; height: 10px; border-radius: 50%; background: var(--ink); }
  .f i.a { background: var(--red); }
  .f button { flex: 1; text-align: left; border: 0; background: var(--soft); border-radius: 10px; padding: 11px 14px; font-size: 16px; }
  .f button.empty { color: var(--muted); }
  .f button.active { box-shadow: inset 0 0 0 2px var(--ink); background: #fff; }
  .swap { position: absolute; right: 14px; top: 50%; margin-top: -15px; width: 36px; height: 36px; border: 0; background: none;
          border-radius: 50%; display: grid; place-items: center; color: var(--muted); }
  .swap:hover { background: var(--soft); color: var(--ink); }
  .ffoot { display: flex; justify-content: space-between; gap: 12px; padding: 8px 20px 12px 42px; font-size: 14px; color: var(--muted); }
  .ffoot button { border: 0; background: none; color: var(--muted); padding: 0; }
  .ffoot button:hover { color: var(--ink); }

  details.params { border-top: 1px solid var(--line); margin-top: 4px; }
  details.params summary { display: flex; justify-content: space-between; padding: 12px 20px; list-style: none; cursor: pointer; font-size: 15px; }
  details.params summary::-webkit-details-marker { display: none; }
  details.params summary b { font-weight: 600; }
  details.params summary b::after { content: ""; display: inline-block; width: 6px; height: 6px; margin: 0 0 3px 8px;
    border-right: 1.5px solid currentColor; border-bottom: 1.5px solid currentColor; transform: rotate(45deg); }
  details.params summary span { color: var(--muted); }
  .row3 { display: grid; grid-template-columns: 1.3fr 1fr 1fr; gap: 8px; padding: 0 20px; }
  .row3 label { font-size: 13px; color: var(--muted); }
  .row3 select, .row3 input { display: block; width: 100%; margin-top: 3px; padding: 7px 8px; border: 1px solid var(--line); border-radius: 8px; background: #fff; font-size: 15px; }
  .params p { margin: 8px 20px 12px; font-size: 14px; color: var(--muted); }

  /* route variants */
  .vars { border-top: 1px solid var(--line); }
  .note { margin: 0; padding: 12px 20px 0; font-size: 14px; color: var(--text); }
  .var { display: block; width: 100%; text-align: left; border: 0; background: none; padding: 16px 20px; border-bottom: 1px solid var(--line); }
  .var:hover { background: #fafbfc; }
  .var.sel { background: var(--soft); box-shadow: inset 4px 0 0 var(--ink); }
  .var .r1 { display: flex; justify-content: space-between; align-items: baseline; }
  .var .r1 b { font-size: 20px; font-weight: 700; }
  .var .r1 span { color: var(--muted); font-size: 15px; }
  .var .sub { color: var(--muted); font-size: 15px; margin-top: 2px; }
  .cdek { display: flex; gap: 10px; align-items: baseline; margin-top: 8px; font-size: 15px; color: var(--text); }
  .badge { flex: none; background: var(--green); color: #fff; font-size: 11px; font-weight: 700; letter-spacing: .02em; border-radius: 5px; padding: 2px 6px; }
  .cdek b { color: var(--ink); }
  .more { margin-top: 10px; font-size: 14px; color: var(--text); display: grid; gap: 3px; }
  .more .save { color: var(--green-ink); font-weight: 600; }
  .link { border: 0; background: none; padding: 0; margin-top: 6px; color: var(--ink); font-weight: 600; font-size: 15px; text-decoration: underline; text-underline-offset: 3px; justify-self: start; }
  .basel { padding: 14px 20px; font-size: 15px; color: var(--text); border-bottom: 1px solid var(--line); }
  .basel b { color: var(--ink); }
  .vgrid { display: grid; grid-template-columns: 1fr 88px; gap: 14px; align-items: center; }
  .vtime { font-size: 24px; font-weight: 750; letter-spacing: -0.02em; line-height: 1.1; }
  .gbox { background: var(--green-strong); color: #fff; border-radius: 10px; padding: 10px 8px; text-align: center; }
  .gbox b { display: block; font-size: 20px; font-weight: 750; line-height: 1.1; letter-spacing: -0.01em; }
  .gbox span { display: block; font-size: 12px; font-weight: 700; letter-spacing: .03em; opacity: .95; }
  .gbox.far { background: #e8eaed; color: var(--ink); }
  .vaddr { font-size: 15px; color: var(--text); margin-top: 6px; }
  .var.plain .r1 b { font-size: 17px; font-weight: 600; color: var(--text); }
  .empty { padding: 16px 20px; color: var(--text); margin: 0; }

  /* notification preview */
  .modal { position: fixed; inset: 0; z-index: 1100; background: rgba(21,24,28,.45); display: grid; place-items: center; }
  .sheet { position: relative; background: #fff; border-radius: 18px; padding: 22px 22px 26px; width: 360px; }
  .sheet h3 { margin: 0 0 14px; font-size: 18px; }
  .close { position: absolute; top: 12px; right: 12px; width: 34px; height: 34px; border: 0; background: var(--soft); border-radius: 50%; display: grid; place-items: center; }
  .phone { width: 300px; margin: 0 auto; border: 1px solid #d6d9de; border-radius: 34px; padding: 10px; background: #f7f7f8; }
  .screen { border-radius: 26px; background: #e9ebee; padding: 16px 10px 40px; }
  .clock { text-align: center; font-size: 34px; font-weight: 300; color: #3b4048; letter-spacing: -0.02em; }
  .date { text-align: center; font-size: 12px; color: var(--muted); margin-bottom: 14px; }
  .push { background: rgba(255,255,255,.95); border-radius: 14px; padding: 10px 12px; font-size: 13px; line-height: 1.35; }
  .push .app { display: flex; align-items: center; gap: 6px; font-size: 11px; color: var(--muted); margin-bottom: 4px; }
  .push .ico { width: 16px; height: 16px; border-radius: 4px; background: var(--green); display: grid; place-items: center; }
  .push .app span:last-child { margin-left: auto; }
  .push b { display: block; margin-bottom: 2px; }

  /* map bits */
  .banner { position: absolute; z-index: 900; bottom: 32px; left: calc(50% + 208px); transform: translateX(-50%);
            background: var(--ink); color: #fff; padding: 8px 14px; border-radius: 8px; font-size: 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,.18); display: none; white-space: nowrap; }
  .banner.on { display: block; }
  .banner.warn { background: #fff; color: var(--amber-ink); border: 1px solid #f0d9a8; }
  .key { position: absolute; z-index: 500; right: 12px; bottom: 28px; background: #fff; border-radius: 10px;
         padding: 8px 12px; font-size: 13px; color: var(--text); box-shadow: 0 1px 4px rgba(0,0,0,.15); }
  .key div { display: flex; align-items: center; gap: 8px; padding: 2px 0; }
  .key i { display: inline-block; width: 22px; height: 0; border-top: 3px solid #4a5058; }
  .key i.ride { border-top: 3px dashed #9aa1aa; }
  .key i.via { border-top: 4px solid var(--green); }
  .key i.dot { width: 10px; height: 10px; border: 2px solid #fff; border-radius: 50%; background: var(--green); margin: 0 6px; box-shadow: 0 0 0 1px #cfd3d8; }
  .leaflet-container { font: inherit; cursor: grab; }
  body.picking .leaflet-container { cursor: crosshair; }
  .m-end { width: 26px; height: 26px; border-radius: 50%; color: #fff; border: 3px solid #fff;
           display: grid; place-items: center; font-weight: 700; font-size: 12px; box-shadow: 0 1px 4px rgba(0,0,0,.35); }
  .m-end.a { background: var(--red); } .m-end.b { background: var(--ink); }
  .m-min { background: var(--green-strong); color: #fff; border-radius: 6px; padding: 2px 7px; font-size: 12px; font-weight: 700;
           white-space: nowrap; box-shadow: 0 1px 3px rgba(0,0,0,.3); transform: translate(10px, -50%); display: inline-block; }
  .m-min.far { background: #fff; color: var(--ink); }

  @media (max-width: 760px) {
    .panel { top: auto; bottom: 0; left: 0; width: 100%; max-height: 62%; border-radius: 16px 16px 0 0; }
    .banner { left: 16px; right: 64px; top: 16px; bottom: auto; transform: none; white-space: normal; text-align: center; }
    .demo { display: none !important; }
    .demo-inline { display: block; }
    .key { display: none; }
  }
</style>
</head>
<body>
<div id="map"></div>
<div class="banner" id="banner"></div>
<div class="demo" id="demo" hidden>
  <div class="dh"><b>Демо-режим</b><small>Готовые маршруты, чтобы быстро посмотреть, как это работает</small></div>
  <div id="demoList"></div>
</div>

<div class="panel" id="panel">
  <div id="vStart">
    <button class="search" id="openPick"><svg width="20" height="20" viewBox="0 0 20 20"><path d="M10 1.5a6 6 0 0 0-6 6c0 4.2 6 11 6 11s6-6.8 6-11a6 6 0 0 0-6-6z" fill="#e5483b"/><circle cx="10" cy="7.5" r="2.2" fill="#fff"/></svg><span>Откуда и куда?</span><svg width="22" height="22" viewBox="0 0 22 22" fill="none" stroke="#3d434b" stroke-width="1.7" stroke-linecap="round"><circle cx="5" cy="5" r="2.3"/><circle cx="17" cy="17" r="2.3"/><path d="M7.3 5H14a3.5 3.5 0 0 1 0 7H8a3.5 3.5 0 0 0 0 7h6.7"/></svg></button>
    <div class="intro">
      <h1>Посылка по пути</h1>
      <p>Постройте маршрут, и мы подскажем пункт СДЭК, куда можно зайти по дороге.</p>
    </div>
    <div class="pad"></div>
  </div>

  <div id="vRoute" hidden>
    <div class="head">
      <button class="back" id="back" aria-label="Назад"><svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M16 10H4M9 5l-5 5 5 5"/></svg></button>
      <span>Маршрут</span>
    </div>
    <div class="fields">
      <div class="f"><i class="a"></i><button data-end="a" id="aBtn"></button></div>
      <div class="f"><i></i><button data-end="b" id="bBtn"></button></div>
      <button class="swap" id="swap" aria-label="Поменять местами"><svg width="18" height="18" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M5 2v12M2 5l3-3 3 3M11 14V2M8 11l3 3 3-3"/></svg></button>
    </div>
    <div class="ffoot"><span id="hint"></span><button id="reset">Сбросить</button></div>

    <div id="pickList"></div>

    <div id="routes" hidden>
      <details class="params" id="params">
        <summary><b>Параметры</b><span id="paramsSum"></span></summary>
        <div class="row3">
          <label>День<select id="day"></select></label>
          <label>Выход<select id="dep"></select></label>
          <label>Посылка<select id="kg">
            <option value="2" selected>до 2 кг</option><option value="5">до 5 кг</option>
            <option value="15">до 15 кг</option><option value="30">до 30 кг</option></select></label>
        </div>
        <p>Пункт должен быть открыт, когда вы до него дойдёте, и принимать посылку такого веса.</p>
      </details>
      <div class="vars" id="variants"></div>
    </div>
  </div>
</div>

<div class="modal" id="modal" hidden>
  <div class="sheet" role="dialog" aria-label="Уведомление">
    <button class="close" id="closeModal" aria-label="Закрыть"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"><path d="M2 2l10 10M12 2 2 12"/></svg></button>
    <h3>Так придёт уведомление</h3>
    <div class="phone"><div class="screen">
      <div class="clock" id="pClock">18:20</div>
      <div class="date" id="pDate"></div>
      <div class="push" id="push"></div>
    </div></div>
  </div>
</div>

<div class="key" id="key" hidden>
  <div><i></i>ваш маршрут</div>
  <div><i class="ride"></i>метро</div>
  <div><i class="via"></i>с заходом в СДЭК</div>
  <div><i class="dot"></i>пункт СДЭК</div>
</div>

<script src="leaflet/leaflet.js"></script>
<script src="data.js"></script>
<script>
"use strict";
const D = window.POTW;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

// ---------- graph ----------
function unpack(b64, T) {
  const bin = atob(b64), buf = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) buf[i] = bin.charCodeAt(i);
  return new T(buf.buffer);
}
const G = {
  n: D.graph.n,
  ll: unpack(D.graph.ll, Int32Array),
  off: unpack(D.graph.off, Uint32Array),
  to: unpack(D.graph.to, Uint32Array),
  w: unpack(D.graph.w, Float32Array),
  eid: unpack(D.graph.eid, Uint32Array),
  geoOff: unpack(D.graph.geoOff, Uint32Array),
  geo: unpack(D.graph.geo, Int32Array),
};
const nodeLL = i => [G.ll[2 * i] / 1e6, G.ll[2 * i + 1] / 1e6];

// The walking graph is symmetric, so one forward pass from B also gives t(P→B).
function dijkstra(src) {
  const dist = new Float64Array(G.n).fill(Infinity);
  const prev = new Int32Array(G.n).fill(-1);
  const pe = new Int32Array(G.n).fill(-1);
  const hn = [], hd = [];
  const push = (n, d) => {
    let i = hn.length; hn.push(n); hd.push(d);
    while (i > 0) { const p = (i - 1) >> 1; if (hd[p] <= d) break;
      hn[i] = hn[p]; hd[i] = hd[p]; i = p; }
    hn[i] = n; hd[i] = d;
  };
  const pop = () => {
    const n = hn[0], d = hd[0], ln = hn.pop(), ld = hd.pop();
    if (hn.length) {
      let i = 0; const m = hn.length;
      for (;;) { let c = 2 * i + 1; if (c >= m) break;
        if (c + 1 < m && hd[c + 1] < hd[c]) c++;
        if (hd[c] >= ld) break;
        hn[i] = hn[c]; hd[i] = hd[c]; i = c; }
      hn[i] = ln; hd[i] = ld;
    }
    return [n, d];
  };
  dist[src] = 0; push(src, 0);
  while (hn.length) {
    const [u, d] = pop();
    if (d !== dist[u]) continue;
    for (let k = G.off[u]; k < G.off[u + 1]; k++) {
      const v = G.to[k], nd = d + G.w[k];
      if (nd < dist[v]) { dist[v] = nd; prev[v] = u; pe[v] = k; push(v, nd); }
    }
  }
  return { dist, prev, pe };
}

function pathLatLngs(run, target) {
  const nodes = [], edges = [];
  for (let v = target; v !== -1; v = run.prev[v]) { nodes.push(v); edges.push(run.pe[v]); }
  nodes.reverse(); edges.reverse();
  const out = [];
  for (let i = 0; i < nodes.length; i++) {
    if (i > 0) {
      const u = nodes[i - 1], v = nodes[i], e = G.eid[edges[i]];
      const s = G.geoOff[e], t = G.geoOff[e + 1], pts = [];
      for (let j = s; j < t; j++) pts.push([G.geo[2 * j] / 1e6, G.geo[2 * j + 1] / 1e6]);
      if (u > v) pts.reverse();
      out.push(...pts);
    }
    out.push(nodeLL(nodes[i]));
  }
  return out;
}

function nearestNode(lat, lon) {
  const k = Math.cos(lat * Math.PI / 180);
  let best = -1, bd = Infinity;
  for (let i = 0; i < G.n; i++) {
    const dy = G.ll[2 * i] / 1e6 - lat, dx = (G.ll[2 * i + 1] / 1e6 - lon) * k;
    const d = dx * dx + dy * dy;
    if (d < bd) { bd = d; best = i; }
  }
  return { node: best, meters: Math.sqrt(bd) * 111320 };
}

const runCache = new Map();
function runFrom(node) {
  if (!runCache.has(node)) {
    if (runCache.size > 12) runCache.delete(runCache.keys().next().value);
    runCache.set(node, dijkstra(node));
  }
  return runCache.get(node);
}

// ---------- time & text helpers ----------
const WD = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun'];
const WD_SHORT = { mon: 'пн', tue: 'вт', wed: 'ср', thu: 'чт', fri: 'пт', sat: 'сб', sun: 'вс' };
const WD_NOM = { mon: 'Понедельник', tue: 'Вторник', wed: 'Среда', thu: 'Четверг', fri: 'Пятница', sat: 'Суббота', sun: 'Воскресенье' };
const WD_FULL = { mon: 'понедельник', tue: 'вторник', wed: 'среда', thu: 'четверг', fri: 'пятница', sat: 'суббота', sun: 'воскресенье' };
const toMin = s => { const [h, m] = s.split(':').map(Number); return h * 60 + m; };
const hhmm = m => { m = Math.round(m) % 1440; return String(Math.floor(m / 60)).padStart(2, '0') + ':' + String(m % 60).padStart(2, '0'); };
const mins = x => x < 0.5 ? '0' : String(Math.round(x));
// detours keep one decimal under 10 min so «+5,2» is not shown as «+5» next to the 5 min threshold
const dmin = x => x < 0.05 ? '0' : x < 10 ? x.toFixed(1).replace('.', ',') : String(Math.round(x));
function plural(n, one, few, many) {
  const a = Math.abs(n) % 100, b = a % 10;
  if (a > 10 && a < 20) return many; if (b > 1 && b < 5) return few; if (b === 1) return one; return many;
}

// same rule as hours.is_open_at: no waiting, handover must finish before closing
function openAt(p, wd, arrive) {
  const h = p.hours[wd];
  if (!h) return { ok: false, why: 'выходной' };
  const o = toMin(h[0]), c = toMin(h[1]), leave = arrive + D.serviceMin;
  if (c < o) {
    if (arrive < o && arrive >= c) return { ok: false, why: `откроется в ${h[0]}` };
    // early-morning part of an overnight window: the handover must end by the close
    if (arrive < c && leave > c) return { ok: false, why: `закроется в ${h[1]}` };
    return { ok: true, close: h[1] };
  }
  if (arrive < o) return { ok: false, why: `откроется в ${h[0]}` };
  if (leave > c) return { ok: false, why: `закроется в ${h[1]}` };
  return { ok: true, close: h[1] };
}

// ---------- state ----------
const placeById = Object.fromEntries(D.places.map(p => [p.id, p]));
// view: 'start' → 'pick' (choosing А/Б) → 'routes'
// mode is always 'metro': buildTrip falls back to walking when that is faster
const S = { view: 'start', a: null, b: null, mode: 'metro', picking: null, selected: null };
const days = [];
for (let d = 12; d <= 18; d++) days.push(new Date(2026, 9, d));
$('day').innerHTML = days.map((d, i) => {
  const wd = WD[(d.getDay() + 6) % 7];
  return `<option value="${i}" ${d.getDate() === 14 ? 'selected' : ''}>${WD_FULL[wd]}</option>`;
}).join('');

$('dep').innerHTML = Array.from({ length: 34 }, (_, i) => {
  const t = String(7 + Math.floor(i / 2)).padStart(2, '0') + ':' + (i % 2 ? '30' : '00');
  return `<option ${t === '18:30' ? 'selected' : ''}>${t}</option>`;
}).join('');

// ---------- map ----------
const map = L.map('map', { zoomControl: false, attributionControl: true });
L.control.zoom({ position: 'topright' }).addTo(map);
// Esri light canvas: calm base so the route and points carry the colour; no API key needed.
// If it fails to load, fall back to Esri World Street Map (the old demo's base).
const ESRI = 'https://server.arcgisonline.com/ArcGIS/rest/services/';
const attr = 'Подложка &copy; Esri, &copy; участники OpenStreetMap';
const base = L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}',
  { maxZoom: 19, maxNativeZoom: 16, attribution: attr }).addTo(map);
const labels = L.tileLayer(ESRI + 'Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
  { maxZoom: 19, maxNativeZoom: 16, pane: 'shadowPane' }).addTo(map);
let tilesOk = 0, tilesBad = 0;
base.on('tileload', () => tilesOk++);
base.on('tileerror', () => {
  if (++tilesBad < 4 || tilesOk) return;
  map.removeLayer(base); map.removeLayer(labels);
  L.tileLayer(ESRI + 'World_Street_Map/MapServer/tile/{z}/{y}/{x}', { maxZoom: 19, attribution: attr, opacity: 0.55 }).addTo(map);
  tilesOk = 1;
});

L.polygon(D.area, { color: '#9aa1aa', weight: 1, dashArray: '4 4', fill: false, interactive: false }).addTo(map);
const routeLayer = L.layerGroup().addTo(map);
const pointLayer = L.layerGroup().addTo(map);
const endIcon = t => L.divIcon({ className: '', html: `<div class="m-end ${t}"></div>`, iconSize: [26, 26], iconAnchor: [13, 13] });
const mA = L.marker([0, 0], { icon: endIcon('a'), draggable: true, zIndexOffset: 1000 });
const mB = L.marker([0, 0], { icon: endIcon('b'), draggable: true, zIndexOffset: 1000 });
mA.on('dragend', () => setEnd('a', mA.getLatLng()));
mB.on('dragend', () => setEnd('b', mB.getLatLng()));
// leave room for the panel on the left
const PANEL_PAD = () => window.innerWidth > 760 ? { paddingTopLeft: [440, 40], paddingBottomRight: [40, 40] }
  : { paddingTopLeft: [20, 70], paddingBottomRight: [20, Math.round(window.innerHeight * 0.62) + 20] };
map.fitBounds(L.latLngBounds(D.area), PANEL_PAD());
map.on('click', e => { if (S.view === 'pick' && S.picking) setEnd(S.picking, e.latlng); });

let bannerTimer = null;
function banner(text, warn) {
  clearTimeout(bannerTimer);
  if (warn) bannerTimer = setTimeout(() => banner(S.view === 'pick' && S.picking ? pickPrompt() : ''), 3000);
  const b = $('banner'); b.textContent = text || ''; b.className = 'banner' + (text ? ' on' : '') + (warn ? ' warn' : '');
}
const pickPrompt = () => `Нажмите на карту, чтобы выбрать «${S.picking === 'a' ? 'Откуда' : 'Куда'}»`;

function setEnd(end, latlng) {
  const snap = nearestNode(latlng.lat, latlng.lng);
  if (snap.meters > 250) {
    banner('Здесь нет данных: прототип покрывает район внутри пунктирной границы', true);
    if (S[end]) (end === 'a' ? mA : mB).setLatLng([S[end].lat, S[end].lon]);
    return;
  }
  S[end] = { id: null, name: 'Точка на карте', gen: 'точки на карте', way: '', lat: latlng.lat, lon: latlng.lng, node: snap.node };
  afterEnd(end);
}
function setPlace(end, id) { S[end] = { ...placeById[id] }; afterEnd(end); }
function afterEnd(end) {
  S.selected = null;
  const other = end === 'a' ? 'b' : 'a';
  if (!S[other]) { go('pick', other); return; }
  go('routes');
}

function go(view, picking) {
  S.view = view; S.picking = view === 'pick' ? (picking || (!S.a ? 'a' : 'b')) : null;
  $('panel').scrollTop = 0;
  render(view === 'routes');
}

// ---------- trip model ----------
function buildTrip() {
  const ra = runFrom(S.a.node), rb = runFrom(S.b.node);
  const walkAll = ra.dist[S.b.node];
  const trip = { legs: [], rides: [] };
  if (S.mode === 'metro') {
    // exit with the shortest walk at each end; same station at both ends → walk
    const pick = run => D.metro.reduce((best, m) => run.dist[m.node] < (best ? run.dist[best.node] : Infinity) ? m : best, null);
    const ma = pick(ra), mb = pick(rb);
    if (ma && mb && ma.station !== mb.station) {
      const hops = Math.abs(D.metroLine.indexOf(ma.station) - D.metroLine.indexOf(mb.station));
      const ride = 5 + 2.5 * hops;
      const legA = { from: S.a, to: { name: 'м. ' + ma.station, gen: 'м. ' + ma.station, node: ma.node }, runA: ra, runB: runFrom(ma.node), start: 0 };
      legA.t = ra.dist[ma.node];
      const legB = { from: { name: 'м. ' + mb.station, node: mb.node }, to: S.b, runA: runFrom(mb.node), runB: rb };
      legB.t = legB.runA.dist[S.b.node];
      legB.start = legA.t + ride;
      if (legA.t + ride + legB.t < walkAll) {
        trip.legs = [legA, legB];
        trip.rides = [{ from: ma, to: mb, t: ride, hops }];
        trip.total = legA.t + ride + legB.t;
        trip.walkAll = walkAll;
        return trip;
      }
      trip.metroSlower = true;
    } else trip.sameStation = true;
  }
  trip.legs = [{ from: S.a, to: S.b, runA: ra, runB: rb, start: 0, t: walkAll }];
  trip.total = walkAll;
  return trip;
}

function scorePoints(trip) {
  const wd = WD[(days[+$('day').value].getDay() + 6) % 7];
  const dep = toMin($('dep').value || '18:30');
  const kg = +$('kg').value;
  return D.points.map(p => {
    // hours are checked on every walking leg first, then the open leg with the smallest detour wins
    let best = null, bestAny = null;
    trip.legs.forEach((leg, i) => {
      const ta = leg.runA.dist[p.node], tb = leg.runB.dist[p.node];
      if (!isFinite(ta) || !isFinite(tb)) return;
      const detour = Math.max(0, ta + tb - leg.t);
      const cand = { detour, leg: i, ta, arrive: dep + leg.start + ta };
      cand.open = openAt(p, wd, cand.arrive);
      if (!bestAny || detour < bestAny.detour) bestAny = cand;
      if (cand.open.ok && (!best || detour < best.detour)) best = cand;
    });
    if (!best) best = bestAny;
    if (!best) return { p, usable: false, why: 'нет пешего пути' };
    const open = best.open;
    const weightOk = kg <= p.maxKg;
    // baseline as in the model: a separate round trip to the point from where you end up (Б)
    const solo = 2 * trip.legs[trip.legs.length - 1].runB.dist[p.node];
    return { p, ...best, solo, open, weightOk, usable: open.ok && weightOk,
             why: !weightOk ? `принимает до ${p.maxKg} кг` : (!open.ok ? open.why : '') };
  }).sort((x, y) => (x.detour ?? 1e9) - (y.detour ?? 1e9));
}

// ---------- render ----------
let trip = null, scored = [];
const ON_WAY_MIN = 5;  // the model's «по пути» threshold
const ICON = {
  place: '<svg width="16" height="16" viewBox="0 0 20 20" fill="currentColor"><path d="M10 2a6 6 0 0 0-6 6c0 4 6 10 6 10s6-6 6-10a6 6 0 0 0-6-6zm0 8.2A2.2 2.2 0 1 1 10 5.8a2.2 2.2 0 0 1 0 4.4z"/></svg>',
  route: '<svg width="16" height="16" viewBox="0 0 22 22" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="5" cy="5" r="2.3"/><circle cx="17" cy="17" r="2.3"/><path d="M7.3 5H14a3.5 3.5 0 0 1 0 7H8a3.5 3.5 0 0 0 0 7h6.7"/></svg>',
};

function render(refit) {
  document.body.classList.toggle('picking', S.view === 'pick');
  $('vStart').hidden = S.view !== 'start';
  $('vRoute').hidden = S.view === 'start';
  $('key').hidden = S.view !== 'routes';
  $('demo').hidden = S.view !== 'pick';
  if (S.view === 'start') {
    S.a = S.b = null; trip = null; scored = [];
    banner('');
  } else {
    for (const end of ['a', 'b']) {
      const btn = $(end + 'Btn');
      btn.textContent = S[end] ? S[end].name : (end === 'a' ? 'Откуда' : 'Куда');
      btn.classList.toggle('empty', !S[end]);
      btn.classList.toggle('active', S.picking === end);
    }
    $('hint').textContent = S.view === 'pick' ? 'Выберите место ниже или на карте' : 'Точки можно перетаскивать на карте';
    $('pickList').hidden = S.view !== 'pick';
    $('routes').hidden = S.view !== 'routes';
    banner(S.view === 'pick' ? pickPrompt() : '');
    if (S.view === 'pick') renderPickList();
  }
  if (S.view === 'routes') {
    trip = buildTrip();
    scored = scorePoints(trip);
    const usable = scored.filter(r => r.usable);
    if (!S.selected || !usable.some(r => r.p.id === S.selected)) S.selected = usable[0]?.p.id ?? null;
    renderParams(); renderVariants(usable);
  }
  renderMap(refit);
}

function renderPickList() {
  const used = new Set([S.a?.id, S.b?.id]);
  $('pickList').innerHTML = '<div class="sh">Места</div>' + D.places.filter(p => !used.has(p.id)).map(p =>
    `<button class="item" data-id="${p.id}"><span class="ic">${ICON.place}</span><span><b>${esc(p.name)}</b><small>${esc(p.hint)}</small></span></button>`).join('')
    + '<div class="demo-inline"><div class="sh sep">Попробуйте демо-маршрут</div>' + presetItems() + '</div>'
    + '<div class="pad"></div>';
  $('pickList').querySelectorAll('.item[data-id]').forEach(b => b.addEventListener('click', () => setPlace(S.picking, b.dataset.id)));
  bindPresets($('pickList'));
}

const presetItems = () => D.presets.map(pr =>
  `<button class="item" data-preset="${pr.id}"><span class="ic">${ICON.route}</span><span><b>${esc(pr.label.replace(/^[^:]+:\s*/, ''))}</b><small>${esc(pr.sub)}</small></span></button>`).join('');
function bindPresets(root) {
  root.querySelectorAll('.item[data-preset]').forEach(b => b.addEventListener('click', () => {
    const pr = D.presets.find(x => x.id === b.dataset.preset);
    S.a = { ...placeById[pr.a] }; S.b = { ...placeById[pr.b] }; S.selected = null;
    go('routes');
  }));
}

function renderParams() {
  const d = days[+$('day').value];
  $('paramsSum').textContent = `${WD_SHORT[WD[(d.getDay() + 6) % 7]]}, выход в ${$('dep').value || '18:30'}, до ${$('kg').value} кг`;
}

function renderVariants(usable) {
  const el = $('variants');
  const dep = toMin($('dep').value || '18:30');
  const base = Math.round(trip.total);
  const parts = [];
  trip.legs.forEach((leg, i) => { parts.push(`пешком ${mins(leg.t)}`); if (trip.rides[i]) parts.push(`метро ${mins(trip.rides[i].t)}`); });
  const how = parts.length > 1 ? parts.join(' + ') + ' мин' : 'пешком';
  const opts = usable.filter(r => r.detour <= D.maxWalkMin).slice(0, 3);
  let html = '';
  if (!opts.length) html += `<p class="empty">Сейчас на этом маршруте нет открытого пункта СДЭК. Попробуйте другое время в параметрах.</p>`;
  else if (opts[0].detour > ON_WAY_MIN) html += `<p class="note">Прямо по пути пунктов нет. Ниже ближайшие и сколько минут добавит заход.</p>`;
  const from = S.b.id === 'home' ? 'из дома' : S.b.id ? `от ${S.b.gen}` : 'от точки Б';
  html = `<div class="basel">Дорога: <b>${base} мин</b>, ${how}, прибытие в ${hhmm(dep + base)}</div>` + html;
  html += opts.map(r => {
    const extra = Math.round(r.detour + D.serviceMin);
    const total = base + extra;
    const sel = r.p.id === S.selected;
    const save = r.solo - r.detour;
    return `<button class="var${sel ? ' sel' : ''}" data-id="${r.p.id}">
      <div class="vgrid">
        <div>
          <div class="vtime">${total} мин</div>
          <div class="sub">прибытие в ${hhmm(dep + total)}</div>
          <div class="vaddr">${esc(r.p.name)}, открыт до ${r.open.close}</div>
        </div>
        <div class="gbox${r.detour <= ON_WAY_MIN ? '' : ' far'}"><span>СДЭК</span><b>+${extra} мин</b></div>
      </div>
      ${sel ? `<div class="more">
        <span>Из ${extra} мин: ${dmin(r.detour)} мин на крюк и ${D.serviceMin} мин на выдачу. В пункте будете в ${hhmm(r.arrive)}.</span>
      </div>` : ''}
    </button>`;
  }).join('');
  el.innerHTML = html;
  el.querySelectorAll('.var[data-id]').forEach(b => b.addEventListener('click', e => {
    if (e.target.closest('[data-push]')) { openPush(); return; }
    S.selected = b.dataset.id; render(false); focusSelected();
  }));
}

function renderMap(refit) {
  routeLayer.clearLayers(); pointLayer.clearLayers();
  for (const [m, end] of [[mA, 'a'], [mB, 'b']]) {
    if (S[end] && S.view !== 'start') { m.setLatLng([S[end].lat, S[end].lon]).addTo(map); } else m.remove();
  }
  const bounds = L.latLngBounds([]); selBounds = null;
  if (S.view === 'routes' && trip) {
    bounds.extend([S.a.lat, S.a.lon]); bounds.extend([S.b.lat, S.b.lon]);
    trip.legs.forEach(leg => {
      const ll = pathLatLngs(leg.runA, leg.to.node);
      ll.forEach(x => bounds.extend(x));
      L.polyline(ll, { color: '#ffffff', weight: 9, opacity: 0.9, interactive: false }).addTo(routeLayer);
      L.polyline(ll, { color: '#4a5058', weight: 5, interactive: false }).addTo(routeLayer);
    });
    trip.rides.forEach(r => {
      L.polyline([[r.from.lat, r.from.lon], [r.to.lat, r.to.lon]], { color: '#9aa1aa', weight: 3, dashArray: '6 6', interactive: false }).addTo(routeLayer);
    });
    const sel = scored.find(r => r.p.id === S.selected);
    if (sel) {
      const leg = trip.legs[sel.leg];
      const ll = pathLatLngs(leg.runA, sel.p.node).concat(pathLatLngs(leg.runB, sel.p.node).reverse());
      L.polyline(ll, { color: '#1ab248', weight: 6, opacity: 0.95, interactive: false }).addTo(routeLayer);
      selBounds = L.latLngBounds(ll); bounds.extend(selBounds);
      L.marker([sel.p.lat, sel.p.lon], { interactive: false, icon: L.divIcon({ className: '', iconSize: null,
        html: `<span class="m-min${sel.detour <= ON_WAY_MIN ? '' : ' far'}">СДЭК +${Math.round(sel.detour + D.serviceMin)} мин</span>` }) }).addTo(pointLayer);
    }
  }
  const byId = Object.fromEntries(scored.map(r => [r.p.id, r]));
  for (const p of D.points) {
    const r = byId[p.id];
    const off = r && !r.usable;
    const m = L.circleMarker([p.lat, p.lon], {
      radius: p.id === S.selected ? 8 : 5, weight: 2, color: '#ffffff',
      fillColor: off ? '#b9bec5' : '#1ab248', fillOpacity: 1,
    }).addTo(pointLayer);
    const type = p.type === 'postamat' ? 'постамат' : 'пункт выдачи';
    const extra = r && isFinite(r.detour ?? Infinity) ? `<br>крюк +${dmin(r.detour)} мин${r.why ? ', ' + esc(r.why) : ''}` : '';
    m.bindTooltip(`<b>${esc(p.name)}</b><br>${type} СДЭК${extra}`, { direction: 'top', offset: [0, -6] });
    if (r && r.usable) m.on('click', () => { S.selected = p.id; render(false); });
  }
  if (refit && bounds.isValid()) map.fitBounds(bounds.pad(0.15), PANEL_PAD());
}

let selBounds = null;
function focusSelected() {
  if (selBounds && !map.getBounds().contains(selBounds)) map.fitBounds(selBounds.extend([S.a.lat, S.a.lon]).extend([S.b.lat, S.b.lon]).pad(0.15), PANEL_PAD());
}

function openPush() {
  const d = days[+$('day').value];
  const wd = WD[(d.getDay() + 6) % 7];
  const dep = toMin($('dep').value || '18:30');
  $('pClock').textContent = hhmm(Math.max(0, dep - 10));
  $('pDate').textContent = WD_NOM[wd];
  const r = scored.find(x => x.p.id === S.selected);
  const title = S.b.way ? `Заберите посылку по дороге ${S.b.way}` : 'Заберите посылку по пути';
  const icon = '<svg width="10" height="10" viewBox="0 0 10 10" fill="#fff"><path d="M5 0 10 2.5v5L5 10 0 7.5v-5z"/></svg>';
  $('push').innerHTML = `<div class="app"><span class="ico">${icon}</span><span>СДЭК</span><span>сейчас</span></div>
    <b>${esc(title)}</b>${esc(r.p.name)}: заход займёт около ${Math.round(r.detour + D.serviceMin)} мин, вы будете там около ${hhmm(r.arrive)}. Пункт открыт до ${r.open.close}.`;
  $('modal').hidden = false;
}

// ---------- controls ----------
$('demoList').innerHTML = presetItems(); bindPresets($('demoList'));
$('openPick').addEventListener('click', () => { S.a = { ...placeById.misis }; S.b = null; go('pick', 'b'); });
$('back').addEventListener('click', () => { go('start'); map.fitBounds(L.latLngBounds(D.area), PANEL_PAD()); });
$('reset').addEventListener('click', () => { S.a = S.b = null; S.selected = null; go('pick', 'a'); });
$('aBtn').addEventListener('click', () => go('pick', 'a'));
$('bBtn').addEventListener('click', () => go('pick', 'b'));
$('swap').addEventListener('click', () => { [S.a, S.b] = [S.b, S.a]; S.selected = null; if (S.a && S.b) go('routes'); else render(false); });
['day', 'dep', 'kg'].forEach(id => $(id).addEventListener('change', () => { render(false); focusSelected(); }));
$('closeModal').addEventListener('click', () => { $('modal').hidden = true; });
$('modal').addEventListener('click', e => { if (e.target === $('modal')) $('modal').hidden = true; });
document.addEventListener('keydown', e => {
  if (e.key !== 'Escape') return;
  if (!$('modal').hidden) { $('modal').hidden = true; return; }
  if (S.view === 'pick' && S.a && S.b) go('routes');
});

render(false);
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
