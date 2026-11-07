"""Load habitual-route config (data/habitual_routes.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "data" / "habitual_routes.yaml"
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


@dataclass
class WalkLeg:
    from_id: str
    to_id: str
    label: str


@dataclass
class Transfer:
    mode: str
    from_id: str
    to_id: str
    label: str
    ride_min: float = 0.0


@dataclass
class HabitualRoute:
    id: str
    label: str
    period: str
    weekdays: tuple[str, ...]
    departure_time: str
    origin: str
    destination: str
    walk_legs: tuple[WalkLeg, ...]
    transfers: tuple[Transfer, ...]

    @property
    def total_ride_min(self) -> float:
        return sum(t.ride_min for t in self.transfers)


@dataclass
class HabitualConfig:
    walk_speed_kmh: float
    service_time_min: float
    example_parcel_weight_kg: float
    morning_range: tuple[str, str]
    evening_range: tuple[str, str]
    routes: tuple[HabitualRoute, ...]
    raw: dict = field(repr=False, default_factory=dict)


def load_habitual_config(path: Path | str = DEFAULT_CONFIG) -> HabitualConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = raw.get("defaults", {})
    ranges = raw.get("experiment_departure_ranges", {})
    morning = ranges.get("morning", {"start": "08:00", "end": "10:30"})
    evening = ranges.get("evening", {"start": "16:00", "end": "21:00"})

    routes = []
    for item in raw.get("habitual_routes", []):
        legs = tuple(
            WalkLeg(leg["from"], leg["to"], leg.get("label", ""))
            for leg in item.get("walk_legs", [])
        )
        transfers = tuple(
            Transfer(
                t["mode"],
                t["from"],
                t["to"],
                t.get("label", ""),
                float(t.get("ride_min", 0.0)),
            )
            for t in item.get("transfers", [])
        )
        weekdays = tuple(item.get("weekdays", ["mon", "tue", "wed", "thu", "fri"]))
        for d in weekdays:
            if d not in WEEKDAYS:
                raise ValueError(f"Unknown weekday {d!r} in route {item.get('id')}")
        routes.append(
            HabitualRoute(
                id=item["id"],
                label=item.get("label", item["id"]),
                period=item["period"],
                weekdays=weekdays,
                departure_time=item["departure_time"],
                origin=item["origin"],
                destination=item["destination"],
                walk_legs=legs,
                transfers=transfers,
            )
        )

    return HabitualConfig(
        walk_speed_kmh=float(defaults.get("walk_speed_kmh", 5.0)),
        service_time_min=float(defaults.get("service_time_min", 4.0)),
        example_parcel_weight_kg=float(defaults.get("example_parcel_weight_kg", 2.0)),
        morning_range=(morning["start"], morning["end"]),
        evening_range=(evening["start"], evening["end"]),
        routes=tuple(routes),
        raw=raw,
    )
