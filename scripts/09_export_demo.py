#!/usr/bin/env python3
"""Export a Leaflet demo page (no build step) from a CDEK points file.

  PYTHONPATH=src .venv/bin/python scripts/09_export_demo.py --csv data/results/points_used.csv

Writes demo/index.html (Leaflet is vendored in demo/leaflet/, so the page opens
straight from disk). Rerun after updating the points.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    dijkstra_with_prev,
    load_graph,
    path_coordinates,
    reconstruct_path,
    snap_to_node,
)
from parcel_on_the_way.points import load_any_points, snap_landmarks, snap_points  # noqa: E402
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402
from parcel_on_the_way.scenarios import (  # noqa: E402
    DetourEngine,
    scenario1_pick_point,
    scenario2_pickup_on_way,
)
from parcel_on_the_way.texts import format_date_ru  # noqa: E402

OUT = ROOT / "demo" / "index.html"
DAY = date(2026, 10, 14)  # Wednesday


def to_lonlat(graph, x, y):
    from pyproj import Transformer

    crs = graph.graph["crs"]
    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(x, y)
    return [round(lon, 6), round(lat, 6)]


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
        src = f"{n} пунктов СДЭК с сайта cdek.ru (часы работы и лимиты веса на 16.10.2026)"
    elif any("osm" in (p.note or "").lower() for p in points):
        src = f"{n} пунктов СДЭК из OpenStreetMap"
    else:
        src = f"{n} пунктов выдачи"
    if unknown:
        src += f"; у {unknown} часы неизвестны и приняты как 10:00–20:00"
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/results/points_used.csv")
    ap.add_argument("--out", default="demo/index.html")
    args = ap.parse_args()

    cfg = load_habitual_config()
    g = load_graph()
    points = load_any_points(ROOT / args.csv)
    snap_points(g, points)
    landmarks_ll = {k: list(v) for k, v in
                    __import__("parcel_on_the_way.graph", fromlist=["LANDMARKS"]).LANDMARKS.items()}
    landmarks = {k: info["node"] for k, info in snap_landmarks(g).items()}

    engine = DetourEngine(g, landmarks, cfg)

    # walking leg geometries (morning + evening unique legs)
    leg_coords = []
    seen = set()
    for route in cfg.routes:
        for leg in route.walk_legs:
            key = (leg.from_id, leg.to_id)
            if key in seen:
                continue
            seen.add(key)
            a, b = landmarks[leg.from_id], landmarks[leg.to_id]
            dist, prev = dijkstra_with_prev(g, a)
            path = reconstruct_path(prev, a, b)
            coords = [to_lonlat(g, x, y) for x, y in path_coordinates(g, path)]
            leg_coords.append({
                "from": leg.from_id,
                "to": leg.to_id,
                "label": leg.label,
                "periods": sorted({r.period for r in cfg.routes
                                   for L in r.walk_legs
                                   if (L.from_id, L.to_id) == key}),
                "minutes": round(dist[b], 1),
                "coords": coords,
            })

    # transit segments (no walking geometry): straight dashed lines between stops
    rides = []
    seen_r = set()
    for route in cfg.routes:
        for t in route.transfers:
            key = tuple(sorted((t.from_id, t.to_id))) + (t.mode,)
            if key in seen_r or t.from_id not in landmarks_ll or t.to_id not in landmarks_ll:
                continue
            seen_r.add(key)
            a, b = landmarks_ll[t.from_id], landmarks_ll[t.to_id]
            rides.append({
                "mode": t.mode,
                "label": {"tram": "трамвай", "metro": "метро"}.get(t.mode, t.mode),
                "minutes": t.ride_min,
                "coords": [[a[1], a[0]], [b[1], b[0]]],
            })

    s1 = scenario1_pick_point(engine, points, day=DAY, parcel_weight_kg=2.0)
    # scenario 2 = the user already chose the scenario-1 winner; show when to pick it up
    s2_point = s1["results"][0]["point_id"] if s1["results"] else points[0].id
    s2 = scenario2_pickup_on_way(
        engine, points, point_id=s2_point, day=DAY, parcel_weight_kg=2.0
    )
    names = {p.id: _display_name(p.name) for p in points}

    pts_json = []
    for p in points:
        top = next((r for r in s1["results"] if r["point_id"] == p.id), None)
        unknown_hours = any(
            p.hours_on(d) == ("10:00", "20:00") for d in ("mon", "sat")
        ) and "unknown" in (p.note or "")
        pts_json.append({
            "id": p.id,
            "name": names[p.id],
            "lat": p.lat,
            "lon": p.lon,
            "type": p.point_type,
            "max_weight_kg": p.max_weight_kg,
            "hours_note": "часы неизвестны → приняты 10:00–20:00" if "unknown" in (p.note or "") else "",
            "reason": top["reason"] if top else None,
            "reason_ru": top["reason_ru"] if top else None,
            "detour_min": top["detour_min"] if top else None,
        })

    landmarks_json = [
        {"id": k, "name": {
            "home": "Дом", "misis": "МИСИС",
            "tram_60letiya": "Трамвай «60-летия Октября»",
            "metro_leninsky_south": "Метро «Ленинский проспект» (южн.)",
            "oktyabrskaya_exit": "Метро «Октябрьская» (выход)",
        }.get(k, k), "lat": v[1], "lon": v[0]}
        for k, v in landmarks_ll.items()
        if k in ("home", "misis", "tram_60letiya", "metro_leninsky_south", "oktyabrskaya_exit")
    ]

    payload = {
        "day_ru": format_date_ru(DAY),
        "legs": leg_coords,
        "rides": rides,
        "points": pts_json,
        "landmarks": landmarks_json,
        "scenario1": [
            {
                "point_id": r["point_id"],
                "name": names[r["point_id"]],
                "reason": r["reason"],
                "reason_ru": r["reason_ru"],
                "detour_min": r["detour_min"],
                "lat": r["lat"],
                "lon": r["lon"],
            }
            for r in s1["results"]
        ],
        "scenario2": {
            "point_id": s2["point_id"],
            "point_name": names.get(s2["point_id"], s2.get("point_name", "")),
            "notification_ru": s2["notification_ru"],
            "ok": s2.get("ok", False),
        },
        "source_ru": _source_ru(points),
    }

    html = TEMPLATE.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out}")
    print(f"points: {len(pts_json)}  scenario1 top: "
          + ", ".join(r["point_id"] for r in s1["results"]))
    print(f"scenario2 ({s2['point_id']}): {s2['notification_ru'][:80]}...")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Посылка по пути — демо</title>
<link rel="stylesheet" href="leaflet/leaflet.css"/>
<style>
  :root { --ink:#1c2430; --muted:#5c6675; --line:#d9d2c5; --accent:#c45c26; --green:#2f8f4e; --blue:#1f6f8b; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:"Segoe UI","PingFang SC",sans-serif; color:var(--ink); background:#f6f3ee; }
  header { padding:16px 20px 8px; max-width:1200px; margin:0 auto; }
  h1 { margin:0 0 4px; font-size:1.35rem; }
  .sub { color:var(--muted); font-size:0.92rem; max-width:80ch; }
  .wrap { max-width:1200px; margin:0 auto; padding:0 20px 28px; display:grid; grid-template-columns:1.4fr 1fr; gap:14px; }
  @media (max-width:960px){ .wrap{ grid-template-columns:1fr; } }
  .card { background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
  #map { height:560px; border-radius:10px; background:#e7eef2; }
  h2 { font-size:1.02rem; margin:0 0 10px; }
  ol { margin:0; padding-left:1.2em; }
  li { margin:0 0 8px; line-height:1.4; }
  .pill { display:inline-block; padding:1px 8px; border-radius:999px; background:#e8f1e9; color:var(--green); font-size:0.8rem; }
  .pill.blue { background:#e5eef2; color:var(--blue); }
  .pill.warn { background:#f8ebe3; color:var(--accent); }
  .note { font-size:0.85rem; color:var(--muted); margin-top:10px; }
  .notif { background:#f3f7f4; border-left:3px solid var(--green); padding:10px 12px; border-radius:8px; margin-top:8px; line-height:1.45; }
  .leg { font-size:0.88rem; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>Посылка по пути — демо</h1>
  <div class="sub">
    Привычные маршруты пользователя (дом ↔ университет МИСИС) и пункты выдачи СДЭК, до которых меньше
    всего идти в сторону от маршрута. День: <b id="day"></b>.
    <div id="source"></div>
  </div>
</header>
<div class="wrap">
  <div class="card">
    <div id="map"></div>
    <div class="note" id="legend"></div>
  </div>
  <div class="card">
    <h2>Сценарий 1 — выбор пункта по пути</h2>
    <ol id="s1"></ol>
    <h2 style="margin-top:16px">Сценарий 2 — забрать по пути</h2>
    <div id="s2"></div>
    <h2 style="margin-top:16px">Пешие участки маршрута</h2>
    <div id="legs"></div>
  </div>
</div>
<script src="leaflet/leaflet.js"></script>
<script>
const DATA = __DATA__;
document.getElementById('day').textContent = DATA.day_ru;
document.getElementById('source').textContent = 'Данные: ' + DATA.source_ru + '.';

const PERIOD_RU = { morning: 'утро', evening: 'вечер' };
const REASON_RU = { detour: 'меньше всего крюк', home: 'ближе всего к дому', open_latest: 'дольше всех открыт' };
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));

const map = L.map('map');
// tile.openstreetmap.org refuses pages opened from disk and CARTO now wants an
// API key (both answer with a picture instead of an error, so no fallback can
// detect it). Esri World Street Map works without a key or Referer.
L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}', {
  maxZoom: 19,
  attribution: 'Подложка &copy; Esri, HERE, Garmin, &copy; участники OpenStreetMap'
}).addTo(map);

const colors = { morning:'#2f8f4e', evening:'#1f6f8b', both:'#c45c26' };
const bounds = L.latLngBounds([]);
for (const leg of DATA.legs) {
  const key = leg.periods.length > 1 ? 'both' : leg.periods[0];
  const latlngs = leg.coords.map(([lon, lat]) => [lat, lon]);
  latlngs.forEach(ll => bounds.extend(ll));
  L.polyline(latlngs, { color: colors[key] || '#888', weight: key === 'evening' ? 4 : 7,
                       opacity: key === 'evening' ? 0.95 : 0.55 })
    .addTo(map).bindPopup(`<b>${esc(leg.label)}</b><br>${leg.minutes} мин`);
}

for (const r of DATA.rides) {
  r.coords.forEach(ll => bounds.extend(ll));
  L.polyline(r.coords, { color: '#5c6675', weight: 3, opacity: 0.8, dashArray: '6 6' })
    .addTo(map).bindTooltip(`${r.label}, ~${r.minutes} мин`, { sticky: true });
}

for (const lm of DATA.landmarks) {
  L.circleMarker([lm.lat, lm.lon], {
    radius: 7, color:'#1c2430', fillColor:'#1c2430', fillOpacity:0.9, weight:1
  }).addTo(map).bindPopup(`<b>${esc(lm.name)}</b>`)
    .bindTooltip(lm.id === 'home' ? 'Дом' : lm.id === 'misis' ? 'МИСИС' : '',
                 { permanent: lm.id === 'home' || lm.id === 'misis', direction: 'left', offset: [-6, 0] });
  bounds.extend([lm.lat, lm.lon]);
}

const topIds = new Set(DATA.scenario1.map(r => r.point_id));
for (const p of DATA.points) {
  const isTop = topIds.has(p.id);
  const color = p.type === 'postamat' ? '#1f6f8b' : '#c45c26';
  const m = L.circleMarker([p.lat, p.lon], {
    radius: isTop ? 10 : 6,
    color: isTop ? '#fff200' : '#ffffff',
    weight: isTop ? 2.5 : 1,
    fillColor: color, fillOpacity: 0.9
  }).addTo(map);
  const typeRu = p.type === 'postamat' ? 'постамат' : 'пункт выдачи';
  let label = `<b>${esc(p.name)}</b><br>${typeRu} СДЭК · посылки до ${p.max_weight_kg} кг`;
  if (p.hours_note) label += `<br><i>${esc(p.hours_note)}</i>`;
  if (p.reason_ru) label += `<br><span class="pill">${esc(p.reason_ru)}</span>`;
  m.bindPopup(label);
  if (isTop) {
    const rank = DATA.scenario1.findIndex(x => x.point_id === p.id) + 1;
    bounds.extend([p.lat, p.lon]);
    const text = String(rank);
    L.marker([p.lat, p.lon], {
      icon: L.divIcon({
        className: '',
        html: `<div style="background:#1c2430;color:#fff;font:600 12px/18px sans-serif;width:18px;text-align:center;border-radius:50%;">${text}</div>`,
        iconSize: null, iconAnchor: rank === 1 ? [-8, 22] : rank === 2 ? [-8, -4] : [26, 9]
      })
    }).addTo(map);
  }
}

const s1 = document.getElementById('s1');
for (const r of DATA.scenario1) {
  const li = document.createElement('li');
  li.innerHTML = `<b>${esc(r.name)}</b> — ${esc(r.reason_ru)} <span class="pill">${REASON_RU[r.reason] || r.reason}</span>`;
  s1.appendChild(li);
}

const s2 = document.getElementById('s2');
s2.innerHTML = `<div class="leg">Выбран пункт: <b>${esc(DATA.scenario2.point_name)}</b></div>
  <div class="notif">${esc(DATA.scenario2.notification_ru)}</div>`;

const legs = document.getElementById('legs');
for (const leg of DATA.legs) {
  const div = document.createElement('div');
  div.className = 'leg';
  const when = leg.periods.map(x => PERIOD_RU[x] || x).join(' и ');
  div.textContent = `${leg.label}: ${leg.minutes} мин (${when})`;
  legs.appendChild(div);
}

document.getElementById('legend').innerHTML =
  'Сплошные линии — пешие участки маршрута: широкая зелёная утром, синяя вечером (часто идут по тем же улицам). ' +
  'Серый пунктир — поездка на трамвае или метро (пешком там не идут, поэтому пункты там не ищем). ' +
  'Чёрные точки — дом, вуз и остановки. Оранжевые кружки — пункты выдачи, синие — постаматы. ' +
  'Жёлтая обводка и номер — три пункта из сценария 1.';
if (bounds.isValid()) map.fitBounds(bounds.pad(0.08)); else map.setView([55.71, 37.595], 13);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
