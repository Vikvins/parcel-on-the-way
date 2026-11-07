"""Load pickup points from CSV and snap them to the walking graph."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .graph import LANDMARKS, snap_to_node

DEFAULT_CSV = Path(__file__).resolve().parents[2] / "data" / "pickup_points.csv"

REQUIRED_COLUMNS = {
    "id", "name", "lat", "lon", "point_type", "max_weight_kg",
    "mon_start", "mon_end", "tue_start", "tue_end", "wed_start", "wed_end",
    "thu_start", "thu_end", "fri_start", "fri_end", "sat_start", "sat_end",
    "sun_start", "sun_end",
}

VALID_TYPES = {"pvz", "postamat"}
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class PickupPoint:
    id: str
    name: str
    lat: float
    lon: float
    point_type: str
    max_weight_kg: float
    hours: dict[str, tuple[str | None, str | None]]  # weekday -> (start, end) or (None, None)
    note: str = ""
    node: int | None = None
    snap_dist_m: float | None = None

    def is_open_on(self, weekday: str) -> bool:
        start, end = self.hours.get(weekday, (None, None))
        return start is not None and end is not None

    def hours_on(self, weekday: str) -> tuple[str | None, str | None]:
        return self.hours.get(weekday, (None, None))


def load_pickup_points(path: Path | str = DEFAULT_CSV) -> list[PickupPoint]:
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")

    points = []
    for rec in frame.to_dict(orient="records"):
        ptype = rec["point_type"].strip().lower()
        if ptype not in VALID_TYPES:
            raise ValueError(f"{rec['id']}: point_type must be one of {sorted(VALID_TYPES)}")
        hours = {}
        for day in WEEKDAYS:
            start = rec[f"{day}_start"].strip()
            end = rec[f"{day}_end"].strip()
            if bool(start) ^ bool(end):
                raise ValueError(f"{rec['id']}: incomplete hours on {day}")
            hours[day] = (start or None, end or None)
        points.append(
            PickupPoint(
                id=rec["id"].strip(),
                name=rec["name"].strip(),
                lat=float(rec["lat"]),
                lon=float(rec["lon"]),
                point_type=ptype,
                max_weight_kg=float(rec["max_weight_kg"]),
                hours=hours,
                note=rec.get("note", "").strip(),
            )
        )

    ids = [p.id for p in points]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate point ids in CSV")
    return points


def snap_points(graph, points: list[PickupPoint]) -> list[PickupPoint]:
    """Fill node / snap_dist_m on each point (mutates and returns the list)."""
    for p in points:
        info = snap_to_node(graph, (p.lon, p.lat))
        p.node = info["node"]
        p.snap_dist_m = info["snap_dist_m"]
    return points


def snap_landmarks(graph) -> dict[str, dict]:
    return {key: snap_to_node(graph, lon_lat) for key, lon_lat in LANDMARKS.items()}


# --- CDEK template format ---------------------------------------------------
# columns: id, address, type, hours_mon..hours_sun ("10:00-21:00"|"closed"),
#          max_weight_kg, source_url, checked_date, lat, lon, geocode_status

CDEK_COLUMNS = {
    "id", "address", "type",
    "hours_mon", "hours_tue", "hours_wed", "hours_thu",
    "hours_fri", "hours_sat", "hours_sun",
}  # optional: max_weight_kg, source_url, checked_date, source, code, lat, lon
DEFAULT_MAX_WEIGHT_KG = 31.0
TYPE_MAP = {
    "пвз": "pvz", "pvz": "pvz",
    "постамат": "postamat", "postamat": "postamat", "почтомат": "postamat",
}


UNKNOWN_HOURS_DEFAULT = ("10:00", "20:00")


def parse_hours_cell(value: str) -> tuple[str | None, str | None]:
    """'10:00-21:00' → ('10:00','21:00'); 'closed'/'' → (None, None).

    'unknown' → default 10:00-20:00 (flagged by callers via note / metadata).
    """
    v = (value or "").strip()
    if v == "" or v.lower() in ("closed", "закрыто", "выходной"):
        return (None, None)
    if v.lower() == "unknown":
        return UNKNOWN_HOURS_DEFAULT
    from .sources import normalize_range

    try:
        return normalize_range(v)
    except ValueError:
        raise ValueError(
            f"bad hours cell: {value!r} (want HH:MM-HH:MM, closed, круглосуточно or unknown)"
        ) from None


def load_cdek_points(path: Path | str) -> list[PickupPoint]:
    """Load the real-CDEK template (address+hours per weekday)."""
    path = Path(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, comment="#")
    # drop instruction/blank rows
    frame = frame[frame["id"].astype(str).str.strip() != ""]
    frame = frame[~frame["id"].astype(str).str.startswith("#")]
    missing = CDEK_COLUMNS.difference(frame.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {sorted(missing)}")

    points = []
    no_coords: list[str] = []
    for rec in frame.to_dict(orient="records"):
        if rec["id"].startswith("EXAMPLE"):
            continue  # skip template example row
        if rec.get("lat", "").strip() == "" or rec.get("lon", "").strip() == "":
            no_coords.append(rec["id"].strip())
            continue
        ptype = TYPE_MAP.get(rec["type"].strip().lower())
        if ptype is None and "постамат" in rec["type"].lower():
            ptype = "postamat"
        if ptype is None:
            raise ValueError(f"{rec['id']}: type must be ПВЗ or постамат, got {rec['type']!r}")
        hours = {}
        for day in WEEKDAYS:
            hours[day] = parse_hours_cell(rec.get(f"hours_{day}", ""))
        max_w = rec.get("max_weight_kg", "").strip()
        points.append(
            PickupPoint(
                id=rec["id"].strip(),
                name=rec.get("address", "").strip() or rec["id"].strip(),
                lat=float(rec["lat"]),
                lon=float(rec["lon"]),
                point_type=ptype,
                max_weight_kg=float(max_w) if max_w else DEFAULT_MAX_WEIGHT_KG,
                hours=hours,
                note=(rec.get("source_url", "") or rec.get("source", "")).strip()
                + (" | hours unknown → 10:00-20:00"
                   if any(rec.get(f"hours_{d}", "").strip().lower() == "unknown"
                          for d in WEEKDAYS) else ""),
            )
        )
    if no_coords:
        import warnings

        msg = (
            f"{path.name}: {len(no_coords)} point(s) without lat/lon skipped "
            f"({', '.join(no_coords[:5])}{'…' if len(no_coords) > 5 else ''}); "
            "geocode them with scripts/06_prepare_cdek_points.py"
        )
        if not points:
            raise ValueError(msg)
        warnings.warn(msg, stacklevel=2)
    return points


def load_any_points(path: Path | str) -> list[PickupPoint]:
    """Auto-detect fictional format vs CDEK template format."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        from .sources import load_real_points

        return load_real_points(path)
    header = pd.read_csv(path, nrows=0, dtype=str, keep_default_na=False, comment="#")
    cols = set(header.columns)
    if "hours_mon" in cols:
        return load_cdek_points(path)
    return load_pickup_points(path)
