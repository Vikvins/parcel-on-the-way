"""Opening-hours checks. Weekday of the scenario is used, not a fixed day."""

from __future__ import annotations

from .points import PickupPoint, WEEKDAYS


def hhmm_to_minutes(hhmm: str) -> int:
    hh, mm = hhmm.split(":")
    return int(hh) * 60 + int(mm)


def minutes_to_hhmm(minutes: float) -> str:
    minutes = int(round(minutes))
    # allow wrapping past midnight for late arrivals
    minutes %= 24 * 60
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def is_open_at(
    point: PickupPoint,
    weekday: str,
    arrival_hhmm: str,
    service_min: float,
) -> tuple[bool, str]:
    """Return (open?, reason). No waiting: arrive before opening → not suitable.

    The point must stay open until arrival + service_min (the handover).
    """
    if weekday not in WEEKDAYS:
        raise ValueError(f"Unknown weekday {weekday!r}")
    start, end = point.hours_on(weekday)
    if start is None or end is None:
        return False, f"closed on {weekday}"

    arrive = hhmm_to_minutes(arrival_hhmm)
    leave = arrive + service_min
    open_m = hhmm_to_minutes(start)
    close_m = hhmm_to_minutes(end)

    if close_m < open_m:
        # overnight window, e.g. 22:00-06:00
        if arrive < open_m and arrive >= close_m:
            return False, f"closed at {arrival_hhmm} (hours {start}-{end})"
        # early-morning part of the window: handover must end by the close
        if arrive < close_m and leave > close_m:
            return False, f"leave {minutes_to_hhmm(leave)} after close {end}"
        # service must not spill past the close of this stretch
        if arrive >= open_m and leave > close_m and leave < open_m:
            return False, f"leave {minutes_to_hhmm(leave)} after close {end}"
        return True, f"open {start}-{end}"

    if arrive < open_m:
        return False, f"arrive {arrival_hhmm} before open {start} (no waiting)"
    if leave > close_m:
        return False, f"leave {minutes_to_hhmm(leave)} after close {end}"
    return True, f"open {start}-{end}"


def weight_ok(point: PickupPoint, parcel_weight_kg: float) -> tuple[bool, str]:
    if parcel_weight_kg <= point.max_weight_kg:
        return True, f"{parcel_weight_kg:g}≤{point.max_weight_kg:g} kg"
    return False, f"{parcel_weight_kg:g}>{point.max_weight_kg:g} kg"
