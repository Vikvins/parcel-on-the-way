"""Readers for real CDEK pickup-point exports.

Supported inputs (auto-detected by `load_real_points`):

* CDEK API v2 ``GET /v2/deliverypoints`` JSON (a list, or ``{"items": [...]}``):
  ``code``, ``type`` (PVZ/POSTAMAT), ``location.latitude/longitude/address``,
  ``work_time_list`` (``[{"day": 1, "time": "10:00/21:00"}, ...]``) or the
  free-text ``work_time``, ``weight_max``, ``is_handout``.
* Overpass JSON (``{"elements": [...]}``) with OSM ``opening_hours`` tags.
* CSV in the CDEK template format (``hours_mon`` … ``hours_sun``), including
  ``data/cdek_real/cdek_moscow_points.csv`` written by the data-collection step.

Every reader returns ``PickupPoint`` objects with per-weekday hours.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .points import (
    DEFAULT_MAX_WEIGHT_KG,
    PickupPoint,
    WEEKDAYS,
    load_cdek_points,
)

Hours = dict[str, tuple[str | None, str | None]]

CLOSED: tuple[None, None] = (None, None)
ALL_DAY = ("00:00", "24:00")

_TIME = r"(\d{1,2})[:.](\d{2})"
_RANGE_RE = re.compile(_TIME + r"\s*[-–—/]\s*" + _TIME)

# --- time ranges --------------------------------------------------------------


def _hhmm(h: str, m: str) -> str:
    return f"{int(h):02d}:{int(m):02d}"


def normalize_range(text: str) -> tuple[str | None, str | None]:
    """'10:00/21:00', '9.00–21.00', 'круглосуточно', '00:00-00:00' → (start, end)."""
    t = (text or "").strip().lower()
    if not t or t in ("closed", "выходной", "закрыто", "off", "-"):
        return CLOSED
    if t in ("24/7", "круглосуточно", "00:00-24:00", "00:00-23:59"):
        return ALL_DAY
    m = _RANGE_RE.search(t)
    if not m:
        raise ValueError(f"cannot parse time range {text!r}")
    start, end = _hhmm(m.group(1), m.group(2)), _hhmm(m.group(3), m.group(4))
    if start == end:  # CDEK writes 24h points as 00:00/00:00
        return ALL_DAY
    if end == "00:00":  # closes at midnight
        end = "24:00"
    return start, end


# --- CDEK API -----------------------------------------------------------------

_RU_DAYS = {"пн": 0, "вт": 1, "ср": 2, "чт": 3, "пт": 4, "сб": 5, "вс": 6}
_EN_DAYS = {"mo": 0, "tu": 1, "we": 2, "th": 3, "fr": 4, "sa": 5, "su": 6}


def _expand_days(spec: str, names: dict[str, int]) -> list[int]:
    """'пн-пт,сб' → [0..5]."""
    days: list[int] = []
    for part in re.split(r"\s*,\s*", spec.strip().lower()):
        if not part:
            continue
        if "-" in part or "–" in part:
            a, b = re.split(r"\s*[-–]\s*", part, maxsplit=1)
            ia, ib = names[a[:2]], names[b[:2]]
            i = ia
            while True:
                days.append(i)
                if i == ib:
                    break
                i = (i + 1) % 7
        else:
            days.append(names[part[:2]])
    return days


def parse_ru_work_time(text: str) -> Hours:
    """CDEK free text 'Пн-Пт 10:00-21:00, Сб-Вс 10:00-18:00' → per-weekday hours."""
    hours: Hours = {d: CLOSED for d in WEEKDAYS}
    t = (text or "").strip()
    if not t:
        raise ValueError("empty work_time")
    if "круглосуточно" in t.lower() and not re.search(r"\d", t):
        return {d: ALL_DAY for d in WEEKDAYS}
    day_word = r"(?:пн|вт|ср|чт|пт|сб|вс)"
    pattern = re.compile(
        rf"({day_word}(?:\s*[-–,]\s*{day_word})*)\s*:?\s*"
        rf"(круглосуточно|выходной|{_TIME}\s*[-–/]\s*{_TIME})",
        re.IGNORECASE,
    )
    found = False
    for m in pattern.finditer(t.lower()):
        found = True
        rng = normalize_range(m.group(2))
        for i in _expand_days(m.group(1), _RU_DAYS):
            hours[WEEKDAYS[i]] = rng
    if not found:
        raise ValueError(f"cannot parse work_time {text!r}")
    return hours


def cdek_api_hours(item: dict) -> tuple[Hours, bool]:
    """Hours from a deliverypoints item. Returns (hours, known)."""
    wtl = item.get("work_time_list") or []
    if wtl:
        hours: Hours = {d: CLOSED for d in WEEKDAYS}
        for rec in wtl:
            day = int(rec["day"])  # 1 = Monday … 7 = Sunday
            hours[WEEKDAYS[day - 1]] = normalize_range(rec.get("time", ""))
        return hours, True
    if item.get("work_time"):
        try:
            return parse_ru_work_time(item["work_time"]), True
        except ValueError:
            pass
    return {d: ("10:00", "20:00") for d in WEEKDAYS}, False


_CDEK_TYPE = {"PVZ": "pvz", "POSTAMAT": "postamat"}


def load_cdek_api_json(path: Path | str) -> list[PickupPoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    items = data.get("items", data) if isinstance(data, dict) else data
    points: list[PickupPoint] = []
    for item in items:
        if item.get("is_handout") is False:
            continue  # reception-only office, parcels are not handed out
        loc = item.get("location") or {}
        lat, lon = loc.get("latitude"), loc.get("longitude")
        if lat is None or lon is None:
            continue
        ptype = _CDEK_TYPE.get(str(item.get("type", "PVZ")).upper(), "pvz")
        hours, known = cdek_api_hours(item)
        max_w = item.get("weight_max")
        address = loc.get("address") or loc.get("address_full") or item.get("name", "")
        points.append(
            PickupPoint(
                id=str(item.get("code") or item.get("uuid")),
                name=f"СДЭК, {address}" if address else str(item.get("code")),
                lat=float(lat),
                lon=float(lon),
                point_type=ptype,
                max_weight_kg=float(max_w) if max_w else DEFAULT_MAX_WEIGHT_KG,
                hours=hours,
                note="cdek_api" + ("" if known else " | hours unknown → 10:00-20:00"),
            )
        )
    return _dedupe(points)


# --- OSM opening_hours --------------------------------------------------------


def parse_osm_opening_hours(value: str) -> Hours:
    """Common subset of OSM opening_hours: '24/7', 'Mo-Fr 10:00-21:00; Sa,Su 10:00-18:00'.

    Raises ValueError on anything it does not understand (PH rules, months…),
    so callers can fall back to the 'unknown' default.
    """
    v = (value or "").strip()
    if not v:
        raise ValueError("empty opening_hours")
    if v == "24/7":
        return {d: ALL_DAY for d in WEEKDAYS}
    hours: Hours = {d: CLOSED for d in WEEKDAYS}
    any_rule = False
    for rule in v.split(";"):
        rule = rule.strip()
        if not rule or rule.upper().startswith("PH"):
            continue
        m = re.fullmatch(
            r"((?:Mo|Tu|We|Th|Fr|Sa|Su)(?:\s*[-,]\s*(?:Mo|Tu|We|Th|Fr|Sa|Su))*)?\s*"
            r"(off|closed|24/7|\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2})",
            rule,
            flags=re.IGNORECASE,
        )
        if not m:
            raise ValueError(f"unsupported opening_hours rule {rule!r}")
        days = (
            _expand_days(m.group(1), _EN_DAYS) if m.group(1) else list(range(7))
        )
        rng = normalize_range(m.group(2))
        for i in days:
            hours[WEEKDAYS[i]] = rng
        any_rule = True
    if not any_rule:
        raise ValueError(f"no rules in {value!r}")
    return hours


_CDEK_NAME_RE = re.compile(r"cdek|сдэк", re.IGNORECASE)


def load_overpass_json(path: Path | str) -> list[PickupPoint]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    points: list[PickupPoint] = []
    for el in data.get("elements", []):
        tags = el.get("tags") or {}
        label = " ".join(tags.get(k, "") for k in ("brand", "name", "operator"))
        if not _CDEK_NAME_RE.search(label):
            continue
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        if lat is None or lon is None:
            continue
        try:
            hours = parse_osm_opening_hours(tags.get("opening_hours", ""))
            note = "osm"
        except ValueError:
            hours = {d: ("10:00", "20:00") for d in WEEKDAYS}
            note = "osm | hours unknown → 10:00-20:00"
        is_locker = (
            tags.get("amenity") == "parcel_locker"
            or "постамат" in label.lower()
            or "postamat" in label.lower()
        )
        street = tags.get("addr:street", "")
        house = tags.get("addr:housenumber", "")
        address = ", ".join(x for x in (street, house) if x)
        points.append(
            PickupPoint(
                id=f"OSM-{el.get('type', 'node')}{el['id']}",
                name=f"СДЭК, {address}" if address else "СДЭК",
                lat=float(lat),
                lon=float(lon),
                point_type="postamat" if is_locker else "pvz",
                max_weight_kg=DEFAULT_MAX_WEIGHT_KG,
                hours=hours,
                note=note,
            )
        )
    return _dedupe(points)


# --- common -------------------------------------------------------------------


def _dedupe(points: list[PickupPoint]) -> list[PickupPoint]:
    seen: set[str] = set()
    out = []
    for p in points:
        if p.id in seen:
            continue
        seen.add(p.id)
        out.append(p)
    return out


def load_real_points(path: Path | str) -> list[PickupPoint]:
    """Load any supported real-CDEK export (JSON from API/Overpass or CSV)."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        head = path.read_text(encoding="utf-8")[:4096]
        if '"elements"' in head:
            return load_overpass_json(path)
        return load_cdek_api_json(path)
    return load_cdek_points(path)


def filter_in_graph(graph, points: list[PickupPoint], max_snap_m: float = 250.0):
    """Snap points; keep those within `max_snap_m` of the walking graph.

    Points far outside the graph area would otherwise snap to its border and
    get fake detours. Vectorised, so a whole-Moscow export (thousands of
    points) is filtered in one pass. Returns (kept, dropped).
    """
    import numpy as np
    import osmnx as ox
    from pyproj import Transformer

    if not points:
        return [], []
    tr = Transformer.from_crs("EPSG:4326", graph.graph["crs"], always_xy=True)
    xs, ys = tr.transform([p.lon for p in points], [p.lat for p in points])
    xs, ys = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    nx_ = np.fromiter((d["x"] for _, d in graph.nodes(data=True)), float)
    ny_ = np.fromiter((d["y"] for _, d in graph.nodes(data=True)), float)
    inside = (
        (xs >= nx_.min() - max_snap_m) & (xs <= nx_.max() + max_snap_m)
        & (ys >= ny_.min() - max_snap_m) & (ys <= ny_.max() + max_snap_m)
    )
    kept: list[PickupPoint] = []
    dropped = [p for p, ok in zip(points, inside) if not ok]
    idx = np.flatnonzero(inside)
    if len(idx):
        nodes, dists = ox.distance.nearest_nodes(
            graph, X=xs[idx], Y=ys[idx], return_dist=True
        )
        for i, node, dist in zip(idx, nodes, dists):
            p = points[i]
            p.node, p.snap_dist_m = int(node), float(dist)
            (kept if dist <= max_snap_m else dropped).append(p)
    return kept, dropped
