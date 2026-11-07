"""Scenario API for the CDEK app: pick a point on the way / pick up on the way.

Both functions return plain JSON-serialisable dicts.
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from .detour import (
    LegDetourResult,
    arrival_offset_min,
    best_leg_for_point,
    commute_timeline,
    detour_for_point,
    score_leg,
)
from .graph import dijkstra_with_prev
from .hours import hhmm_to_minutes, is_open_at, minutes_to_hhmm, weight_ok
from .points import PickupPoint
from .routes_config import HabitualConfig, WEEKDAYS
from .texts import (
    format_close_ru,
    format_date_ru,
    format_detour_ru,
    format_notification_detour,
    format_period_ru,
    format_weight_rejection,
)

STORAGE_DAYS_DEFAULT = 7

# weekday name from date
_WD = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def weekday_of(day: date) -> str:
    return _WD[day.weekday()]


def _date_from(value: str | date) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(value)


def _departure_minutes(hhmm: str) -> int:
    return hhmm_to_minutes(hhmm)


def _close_minutes(point: PickupPoint, weekday: str) -> int | None:
    end = point.hours_on(weekday)[1]
    return None if end is None else hhmm_to_minutes(end)


def _open_minutes(point: PickupPoint, weekday: str) -> int | None:
    start = point.hours_on(weekday)[0]
    return None if start is None else hhmm_to_minutes(start)


class DetourEngine:
    """Precompute two-Dijkstra boards for every walk leg of every commute.

    Reuses boards per (commute_id, weekday-independent) leg so scenario calls
    are cheap. Leg geometry does not depend on weekday — only hours do.
    """

    def __init__(self, graph, landmarks: dict[str, int], cfg: HabitualConfig):
        self.graph = graph
        self.landmarks = landmarks
        self.cfg = cfg
        self._boards: dict[tuple[str, str, str], Any] = {}
        # home → every node, for "closest to home"
        self._home_dist, _ = dijkstra_with_prev(graph, landmarks["home"])

    def board(self, route_id: str, leg_from: str, leg_to: str):
        key = (route_id, leg_from, leg_to)
        if key not in self._boards:
            route = next(r for r in self.cfg.routes if r.id == route_id)
            self._boards[key] = score_leg(self.graph, leg_from, leg_to, self.landmarks)
        return self._boards[key]

    def home_walk_min(self, point: PickupPoint) -> float:
        return self._home_dist.get(point.node, math.inf)


def _evaluate_point_on_commute(
    engine: DetourEngine,
    route,
    point: PickupPoint,
    weekday: str,
    day: date,
    parcel_weight_kg: float,
) -> dict | None:
    """Score one point on one commute for one weekday. None if unreachable."""
    events = commute_timeline(route)
    leg_duration = {}
    boards = []
    for leg in route.walk_legs:
        b = engine.board(route.id, leg.from_id, leg.to_id)
        boards.append(b)
        leg_duration[(leg.from_id, leg.to_id)] = b.t_a_to_b_min
    for ev in events:
        if ev["kind"] == "walk":
            ev["leg_duration_min"] = leg_duration.get((ev["from"], ev["to"]), 0.0)

    scored = [detour_for_point(b, point) for b in boards]
    scored = [s for s in scored if not math.isinf(s.detour_min)]
    if not scored:
        return None

    # Opening hours are checked on every walking leg first; the open leg with
    # the smallest detour wins. Only if no leg is open, the smallest detour is
    # reported (as not usable), so a point open on another leg is never lost.
    dep = _departure_minutes(route.departure_time)
    per_leg = []
    for s in scored:
        offset = arrival_offset_min(events, s.leg_from, s.leg_to, s.t_a_to_p_min)
        arrive_min = dep + offset
        arrive_hhmm = minutes_to_hhmm(arrive_min)
        ok, why = is_open_at(point, weekday, arrive_hhmm, engine.cfg.service_time_min)
        per_leg.append((s, arrive_min, arrive_hhmm, ok, why))
    open_legs = [x for x in per_leg if x[3]]
    best, arrive_min, arrive_hhmm, open_ok, open_reason = min(
        open_legs or per_leg, key=lambda x: (x[0].detour_min, x[1])
    )
    wt_ok, wt_reason = weight_ok(point, parcel_weight_kg)
    close_m = _close_minutes(point, weekday)

    period_ru = "из дома в вуз" if route.period == "morning" else "из вуза домой"
    return {
        "point_id": point.id,
        "point_name": point.name,
        "point_type": point.point_type,
        "lat": point.lat,
        "lon": point.lon,
        "commute_id": route.id,
        "commute_label": route.label,
        "period": route.period,
        "period_ru": period_ru,
        "leg_from": best.leg_from,
        "leg_to": best.leg_to,
        "leg_label": f"{best.leg_from}→{best.leg_to}",
        "t_a_to_p_min": round(best.t_a_to_p_min, 2),
        "t_p_to_b_min": round(best.t_p_to_b_min, 2),
        "t_a_to_b_min": round(best.t_a_to_b_min, 2),
        "detour_min": round(best.detour_min, 2),
        "on_shortest_path": best.on_shortest_path,
        "arrival": arrive_hhmm,
        "arrival_minutes": arrive_min,
        "departure": route.departure_time,
        "weekday": weekday,
        "date": day.isoformat(),
        "open": open_ok,
        "open_reason": open_reason,
        "hours_today": (
            None
            if point.hours_on(weekday)[0] is None
            else f"{point.hours_on(weekday)[0]}-{point.hours_on(weekday)[1]}"
        ),
        "close_hhmm": minutes_to_hhmm(close_m) if close_m is not None else None,
        "close_minutes": close_m,
        "weight_ok": wt_ok,
        "weight_reason": wt_reason,
        "usable": open_ok and wt_ok,
        "home_walk_min": (
            None
            if math.isinf(engine.home_walk_min(point))
            else round(engine.home_walk_min(point), 2)
        ),
    }


def _explain(r: dict, reason: str) -> str:
    detour_s = format_detour_ru(r["detour_min"])
    if r["detour_min"] >= 0.5:
        detour_s = f"крюк {detour_s}"
    close_s = format_close_ru(r["close_hhmm"], r.get("hours_today"))
    if reason == "detour":
        head = (
            f"прямо по пути {r['period_ru']}"
            if r["detour_min"] < 0.5
            else f"по дороге {r['period_ru']}, {detour_s}"
        )
    elif reason == "home":
        home_min = r["home_walk_min"]
        home_part = (
            f"{home_min:.0f} мин от дома" if home_min is not None else "близко к дому"
        )
        head = f"ближе всего к дому ({home_part}), {detour_s}"
    elif reason == "open_latest":
        if r.get("hours_today") in ("00:00-23:59", "00:00-24:00") or r["close_hhmm"] in (
            "23:59",
            "24:00",
        ):
            head = f"работает круглосуточно, {detour_s}"
        else:
            head = f"дольше всех открыт (до {r['close_hhmm']}), {detour_s}"
    else:
        head = detour_s
    if r["open"] and reason != "open_latest":
        if close_s == "круглосуточно":
            head += ", круглосуточно"
        elif r["close_hhmm"]:
            head += f", открыт до {r['close_hhmm']}"
    return head


# ---------------------------------------------------------------------------
# Scenario 1
# ---------------------------------------------------------------------------

def scenario1_pick_point(
    engine: DetourEngine,
    points: list[PickupPoint],
    day: str | date = "2026-10-14",
    parcel_weight_kg: float = 2.0,
    top_k: int = 3,
) -> dict[str, Any]:
    """Top-3 pickup points on the way, with distinct reasons.

    A point is a candidate if it is usable on AT LEAST ONE commute that day.
    Best commute for each candidate = smallest detour among usable ones.
    """
    day = _date_from(day)
    weekday = weekday_of(day)

    candidates: list[dict] = []
    considered: list[dict] = []
    for p in points:
        options = []
        for route in engine.cfg.routes:
            if weekday not in route.weekdays:
                continue
            ev = _evaluate_point_on_commute(
                engine, route, p, weekday, day, parcel_weight_kg
            )
            if ev is None:
                continue
            options.append(ev)
        if not options:
            continue
        usable = [ev for ev in options if ev["usable"]]
        considered.extend(options)
        if not usable:
            # keep for diagnostics: not a candidate
            continue
        best = min(usable, key=lambda ev: (ev["detour_min"], ev["arrival_minutes"]))
        candidates.append(best)

    def pick(remaining: list[dict], reason: str) -> tuple[dict | None, list[dict]]:
        if not remaining:
            return None, remaining
        if reason == "detour":
            winner = min(remaining, key=lambda r: r["detour_min"])
        elif reason == "home":
            winner = min(
                remaining,
                key=lambda r: (r["home_walk_min"] is None, r["home_walk_min"] or 1e9),
            )
        elif reason == "open_latest":
            winner = max(
                remaining,
                key=lambda r: (
                    r["close_minutes"] is not None,
                    r["close_minutes"] or -1,
                    -r["detour_min"],
                ),
            )
        else:
            winner = remaining[0]
        rest = [r for r in remaining if r["point_id"] != winner["point_id"]]
        item = dict(winner)
        item["reason"] = reason
        item["reason_ru"] = _explain(winner, reason)
        return item, rest

    selected = []
    remaining = list(candidates)
    for reason in ("detour", "home", "open_latest"):
        item, remaining = pick(remaining, reason)
        if item is not None:
            selected.append(item)

    return {
        "scenario": "pick_point_on_the_way",
        "date": day.isoformat(),
        "weekday": weekday,
        "parcel_weight_kg": parcel_weight_kg,
        "service_time_min": engine.cfg.service_time_min,
        "n_candidates": len(candidates),
        "n_points_considered": len(points),
        "results": selected,
        "candidates": sorted(candidates, key=lambda r: r["detour_min"]),
    }


# ---------------------------------------------------------------------------
# Scenario 2
# ---------------------------------------------------------------------------

def scenario2_pickup_on_way(
    engine: DetourEngine,
    points: list[PickupPoint],
    point_id: str,
    day: str | date = "2026-10-14",
    parcel_weight_kg: float = 2.0,
    storage_days: int = STORAGE_DAYS_DEFAULT,
) -> dict[str, Any]:
    """Pick the best commute/leg to stop by a fixed point on a given date.

    If nothing works that day, suggest the earliest usable option within
    `storage_days` (default 7 — typical CDEK storage).
    """
    day = _date_from(day)
    by_id = {p.id: p for p in points}
    if point_id not in by_id:
        return {
            "scenario": "pickup_on_the_way",
            "ok": False,
            "error": f"unknown point_id {point_id!r}",
        }
    point = by_id[point_id]

    # Weight is a property of the point — no need to scan other days.
    if parcel_weight_kg > point.max_weight_kg:
        return {
            "scenario": "pickup_on_the_way",
            "ok": False,
            "point_id": point.id,
            "point_name": point.name,
            "date": day.isoformat(),
            "weekday": weekday_of(day),
            "parcel_weight_kg": parcel_weight_kg,
            "chosen": None,
            "fallback": None,
            "reason": "weight",
            "reason_ru": (
                f"посылка {parcel_weight_kg:g} кг тяжелее лимита "
                f"{point.max_weight_kg:g} кг"
            ),
            "notification_ru": format_weight_rejection(
                point.name, parcel_weight_kg, point.max_weight_kg
            ),
        }

    def best_for(day_i: date) -> dict | None:
        wd = weekday_of(day_i)
        options = []
        for route in engine.cfg.routes:
            if wd not in route.weekdays:
                continue
            ev = _evaluate_point_on_commute(
                engine, route, point, wd, day_i, parcel_weight_kg
            )
            if ev is not None and ev["usable"]:
                options.append(ev)
        if not options:
            return None
        return min(options, key=lambda ev: (ev["detour_min"], ev["arrival_minutes"]))

    same_day = best_for(day)
    if same_day is not None:
        notification = format_notification_detour(
            same_day["period"],
            same_day["detour_min"],
            same_day["close_hhmm"],
            same_day["hours_today"],
        )
        return {
            "scenario": "pickup_on_the_way",
            "ok": True,
            "point_id": point.id,
            "point_name": point.name,
            "date": day.isoformat(),
            "weekday": weekday_of(day),
            "parcel_weight_kg": parcel_weight_kg,
            "chosen": same_day,
            "notification_ru": notification,
            "fallback": None,
        }

    # fallback: earliest usable day within the storage window
    fallback = None
    for k in range(1, storage_days + 1):
        d = day + timedelta(days=k)
        ev = best_for(d)
        if ev is not None:
            fallback = ev
            break

    wd = weekday_of(day)
    active_routes = [r for r in engine.cfg.routes if wd in r.weekdays]
    if not active_routes:
        why = "привычные маршруты только по будням"
    else:
        options = []
        for route in active_routes:
            ev = _evaluate_point_on_commute(
                engine, route, point, wd, day, parcel_weight_kg
            )
            if ev is not None:
                options.append(ev)
        if not options:
            why = "нет пеших ног до пункта в привычных маршрутах"
        else:
            closed = [ev for ev in options if not ev["open"]]
            why = "не успеваете до закрытия" if closed else "пункт недоступен"

    date_s = format_date_ru(day, wd)
    if fallback is None:
        return {
            "scenario": "pickup_on_the_way",
            "ok": False,
            "point_id": point.id,
            "point_name": point.name,
            "date": day.isoformat(),
            "weekday": wd,
            "parcel_weight_kg": parcel_weight_kg,
            "chosen": None,
            "fallback": None,
            "reason_ru": why,
            "notification_ru": (
                f"{date_s.capitalize()} забрать не получится ({why}). "
                f"В ближайшие {storage_days} дн. удобного варианта тоже нет."
            ),
        }

    fb_day = date.fromisoformat(fallback["date"])
    fb_date = format_date_ru(fb_day, fallback["weekday"])
    notification = (
        f"{date_s.capitalize()} не получится ({why}). "
        f"Ближайший удобный вариант — {fb_date}, "
        f"{format_period_ru(fallback['period'])}, "
        f"{'крюк ' if fallback['detour_min'] >= 0.5 else ''}{format_detour_ru(fallback['detour_min'])}. "
        f"{format_close_ru(fallback['close_hhmm'], fallback.get('hours_today')).capitalize()}."
    )
    return {
        "scenario": "pickup_on_the_way",
        "ok": True,
        "point_id": point.id,
        "point_name": point.name,
        "date": day.isoformat(),
        "weekday": wd,
        "parcel_weight_kg": parcel_weight_kg,
        "chosen": None,
        "fallback": fallback,
        "reason_ru": why,
        "notification_ru": notification,
    }
