#!/usr/bin/env python3
"""Pull CDEK pickup points / lockers from OSM in the large study area.

Writes a draft CSV in the template format with OSM ids, for manual verification
against the CDEK site. Counts per rough district.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import study_area_box  # noqa: E402

UA = {"User-Agent": "parcel-on-the-way-prototype/0.1 (student research)"}
OUT = ROOT / "data" / "cdek_points_osm_draft.csv"

DISTRICTS = {
    "Якиманка": (55.720, 37.590, 55.755, 37.655),
    "Хамовники": (55.710, 37.540, 55.750, 37.600),
    "Донской": (55.690, 37.570, 55.720, 37.630),
    "Гагаринский": (55.680, 37.540, 55.710, 37.595),
    "Академический": (55.665, 37.540, 55.700, 37.595),
}


def district_of(lat: float, lon: float) -> str:
    for name, (y0, x0, y1, x1) in DISTRICTS.items():
        if y0 <= lat <= y1 and x0 <= lon <= x1:
            return name
    return "другое / граница"


def run_query(q: str):
    eps = [
        "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    for ep in eps:
        for attempt in range(3):
            try:
                r = requests.post(ep, data={"data": q}, headers=UA, timeout=120)
                if r.status_code == 200:
                    return r.json().get("elements", [])
                print(f"  {ep.split('/')[2]} status={r.status_code}")
            except Exception as e:
                print(f"  {ep.split('/')[2]} ERR {e}")
            time.sleep(3)
    return []


def main():
    bounds, _ = study_area_box(large=True)
    print(f"Bbox: {bounds}")

    q = f"""
[out:json][timeout:120];
(
  nwr["brand"~"CDEK|СДЭК",i]({bounds[1]:.5f},{bounds[0]:.5f},{bounds[3]:.5f},{bounds[2]:.5f});
  nwr["name"~"CDEK|СДЭК|Сдэк",i]({bounds[1]:.5f},{bounds[0]:.5f},{bounds[3]:.5f},{bounds[2]:.5f});
  nwr["operator"~"CDEK|СДЭК",i]({bounds[1]:.5f},{bounds[0]:.5f},{bounds[3]:.5f},{bounds[2]:.5f});
);
out center tags;
"""
    els = run_query(q)
    print(f"OSM elements: {len(els)}")

    rows = []
    seen = set()
    counts: dict[str, int] = {}
    for el in els:
        tags = el.get("tags") or {}
        lat = el.get("lat") or (el.get("center") or {}).get("lat")
        lon = el.get("lon") or (el.get("center") or {}).get("lon")
        if lat is None or lon is None:
            continue
        key = (round(float(lat), 6), round(float(lon), 6))
        if key in seen:
            continue
        seen.add(key)
        name = tags.get("name") or tags.get("brand") or "CDEK"
        addr_parts = [
            tags.get("addr:street", ""),
            tags.get("addr:housenumber", ""),
        ]
        address = ", ".join(x for x in addr_parts if x) or f"{lat:.5f}, {lon:.5f}"
        ptype = "постамат" if tags.get("amenity") == "parcel_locker" or "постамат" in name.lower() else "ПВЗ"
        dist = district_of(float(lat), float(lon))
        counts[dist] = counts.get(dist, 0) + 1
        rows.append({
            "id": f"OSM-{el.get('type','n')}{el['id']}",
            "address": address,
            "type": ptype,
            "hours_mon": "", "hours_tue": "", "hours_wed": "", "hours_thu": "",
            "hours_fri": "", "hours_sat": "", "hours_sun": "",
            "max_weight_kg": "",
            "source_url": f"https://www.openstreetmap.org/{el.get('type','node')}/{el['id']}",
            "checked_date": "",
            "lat": f"{float(lat):.7f}",
            "lon": f"{float(lon):.7f}",
            "geocode_status": "from_osm",
            "osm_name": name,
            "district": dist,
            "note": "draft from OSM — verify hours/weight on cdek.ru",
        })

    import csv

    fields = [
        "id", "address", "type",
        "hours_mon", "hours_tue", "hours_wed", "hours_thu",
        "hours_fri", "hours_sat", "hours_sun",
        "max_weight_kg", "source_url", "checked_date",
        "lat", "lon", "geocode_status", "osm_name", "district", "note",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {OUT} ({len(rows)} points)")

    print("\nPer district:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {k:20s} {v}")
    print("\nFill hours_* and max_weight_kg from cdek.ru, then run 06_prepare + 05_experiment.")


if __name__ == "__main__":
    main()
