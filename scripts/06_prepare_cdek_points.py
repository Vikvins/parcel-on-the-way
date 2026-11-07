#!/usr/bin/env python3
"""Prepare real CDEK points: geocode addresses, validate hours, draw check map.

  PYTHONPATH=src .venv/bin/python scripts/06_prepare_cdek_points.py \
      --csv data/cdek_points.csv

Fills lat/lon/geocode_status in the SAME csv. Caches geocoder answers in
data/raw/geocode_cache.json (1 req/sec to Nominatim). Flags rows outside the
study bbox or unparseable hours. Then draws a map for visual check.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.graph import (  # noqa: E402
    LANDMARKS,
    load_graph,
    project_lon_lat,
    snap_to_node,
    study_area_box,
)
from parcel_on_the_way.points import parse_hours_cell  # noqa: E402

UA = {"User-Agent": "parcel-on-the-way-prototype/0.1 (student research)"}
NOMINATIM = "https://nominatim.openstreetmap.org/search"
CACHE_PATH = ROOT / "data" / "raw" / "geocode_cache.json"
PAUSE_S = 1.1
HOURS_RE = re.compile(r"^([01]?\d|2[0-3]):[0-5]\d-([01]?\d|2[0-3]):[0-5]\d$")


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def geocode(address: str, cache: dict) -> tuple[float, float] | None:
    key = address.strip().lower()
    if key in cache:
        val = cache[key]
        return None if val is None else (float(val["lat"]), float(val["lon"]))
    time.sleep(PAUSE_S)
    r = requests.get(
        NOMINATIM,
        params={"q": address, "format": "json", "limit": 1, "countrycodes": "ru"},
        headers=UA,
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    if not data:
        cache[key] = None
        save_cache(cache)
        return None
    lat, lon = float(data[0]["lat"]), float(data[0]["lon"])
    cache[key] = {"lat": lat, "lon": lon, "display_name": data[0].get("display_name", "")}
    save_cache(cache)
    return lat, lon


def in_bbox(lon: float, lat: float, bounds) -> bool:
    return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/cdek_points.csv")
    ap.add_argument("--map", default="data/processed/cdek_points_check_map.png")
    args = ap.parse_args()

    path = ROOT / args.csv if not Path(args.csv).is_absolute() else Path(args.csv)
    if not path.exists():
        print(f"File not found: {path}")
        print("Copy data/cdek_points_template.csv to data/cdek_points.csv and fill it.")
        sys.exit(1)

    frame = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    frame = frame[frame["id"].astype(str).str.strip() != ""]
    frame = frame[~frame["id"].astype(str).str.startswith("#")]

    bounds, _ = study_area_box()
    print(f"Study bbox: lon[{bounds[0]:.4f},{bounds[2]:.4f}] lat[{bounds[1]:.4f},{bounds[3]:.4f}]")
    print(f"Rows to check: {len(frame)}")

    cache = load_cache()
    statuses = []
    for i, rec in enumerate(frame.to_dict(orient="records")):
        issues = []
        # hours format
        for day in ("mon", "tue", "wed", "thu", "fri", "sat", "sun"):
            cell = rec.get(f"hours_{day}", "").strip()
            if cell == "" or cell.lower() in ("closed", "закрыто"):
                continue
            if not HOURS_RE.match(cell):
                issues.append(f"bad hours_{day}={cell!r}")
        # geocode if needed
        lat_s, lon_s = rec.get("lat", "").strip(), rec.get("lon", "").strip()
        if lat_s and lon_s:
            lat, lon = float(lat_s), float(lon_s)
            status = "ok"
        else:
            addr = rec.get("address", "").strip()
            if not addr:
                status = "no_address"
                lat, lon = float("nan"), float("nan")
                issues.append("empty address")
            else:
                got = geocode(addr, cache)
                if got is None:
                    status = "not_found"
                    lat, lon = float("nan"), float("nan")
                    issues.append("geocoder: not found")
                else:
                    lat, lon = got
                    status = "ok"
        if status == "ok" and not in_bbox(lon, lat, bounds):
            status = "outside_area"
            issues.append(f"outside study bbox ({lon:.5f},{lat:.5f})")
        # write back
        frame.iloc[i, frame.columns.get_loc("lat")] = "" if pd.isna(lat) else f"{lat:.7f}"
        frame.iloc[i, frame.columns.get_loc("lon")] = "" if pd.isna(lon) else f"{lon:.7f}"
        if "geocode_status" in frame.columns:
            frame.iloc[i, frame.columns.get_loc("geocode_status")] = status
        statuses.append((rec["id"], status, issues, lat, lon))
        flag = "OK " if status == "ok" and not issues else "!! "
        print(f"  {flag}{rec['id']:20s} {status:16s} {lat:.5f} {lon:.5f}  {'; '.join(issues)}")

    frame.to_csv(path, index=False)
    print(f"\nUpdated {path}")

    # map
    try:
        g = load_graph()
    except FileNotFoundError:
        print("Graph not built — skip map")
        return

    out = ROOT / args.map
    fig, ax = plt.subplots(figsize=(10, 10))
    for u, v, _k, data in g.edges(keys=True, data=True):
        geom = data.get("geometry")
        if geom is None:
            continue
        c = list(geom.coords)
        ax.plot([p[0] for p in c], [p[1] for p in c], color="#e2e8ee",
                linewidth=0.3, zorder=1)

    for key, lon_lat in LANDMARKS.items():
        if key.startswith("tram_") and key not in ("tram_60letiya",):
            continue
        x, y = project_lon_lat(g, lon_lat)
        ax.scatter([x], [y], s=40, color="#1c2430", zorder=4)
        ax.annotate(key, (x, y), textcoords="offset points", xytext=(6, 4),
                    fontsize=7, color="#1c2430")

    for rec_id, status, issues, lat, lon in statuses:
        if status != "ok" or pd.isna(lat):
            continue
        x, y = project_lon_lat(g, (lon, lat))
        color = "#2f8f4e" if not issues else "#c45c26"
        ax.scatter([x], [y], s=70, color=color, marker="s", zorder=5,
                   edgecolors="white")
        ax.annotate(rec_id, (x, y), textcoords="offset points", xytext=(8, -10),
                    fontsize=7.5)

    ax.set_aspect("equal")
    ax.set_title("Проверка пунктов СДЭК (зелёный = ок, оранжевый = есть замечания)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Map: {out}")

    ok = sum(1 for _, s, iss, _, _ in statuses if s == "ok" and not iss)
    print(f"\nSummary: {ok}/{len(statuses)} fully OK")


if __name__ == "__main__":
    main()
