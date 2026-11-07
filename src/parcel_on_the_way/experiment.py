"""Experiment: 3 methods for choosing a pickup point, same cost metric.

Methods
-------
baseline_straight : closest point to home in a straight line
baseline_network  : closest point to home along the walking network
ours              : smallest usable detour across the user's commutes

Cost (extra walking time vs the habitual legs, with arrival + service check)
----------------------------------------------------------------------------
- If the point is usable on some commute and "on the way" (detour ≤
  on_the_way_max_min): cost = detour.
- Else (baselines often land here): cost = round trip home → point → home
  (a separate trip), also returned as round_trip_home_min.

NOTE: results on the 10 fictional points are a pipeline check only.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any

import networkx as nx

from .detour import detour_for_point, score_leg
from .graph import MAX_WALK_MIN, dijkstra_with_prev
from .hours import hhmm_to_minutes, is_open_at, minutes_to_hhmm, weight_ok
from .points import PickupPoint
from .routes_config import HabitualConfig, WalkLeg, Transfer, HabitualRoute
from .texts import format_date_ru, format_detour_ru, format_period_ru

SEED = 20261015
ON_THE_WAY_MAX_MIN = 10.0
MIN_OD_STRAIGHT_M = 1500.0
DEFAULT_RIDE_WAIT_MIN = 3.0
DEFAULT_RIDE_MIN_PER_KM = 2.0  # ≈ 30 km/h average incl. stops

WD_EN = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


@dataclass
class TransitStop:
    osm_id: int
    lat: float
    lon: float
    name: str
    kind: str  # metro_exit | tram_stop | tram_platform
    node: int | None = None
    snap_dist_m: float | None = None


@dataclass
class SyntheticUser:
    user_id: str
    home_node: int
    dest_node: int
    home_stop: TransitStop
    dest_stop: TransitStop
    ride_min: float
    weekday: str
    morning_dep: str
    evening_dep: str
    routes: tuple[HabitualRoute, ...]
    is_real: bool = False
    label: str = ""


def load_transit_stops(path) -> list[TransitStop]:
    import csv

    stops = []
    with open(path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            stops.append(
                TransitStop(
                    osm_id=int(rec["osm_id"]),
                    lat=float(rec["lat"]),
                    lon=float(rec["lon"]),
                    name=rec["name"],
                    kind=rec["kind"],
                )
            )
    return stops


def snap_stops(graph, stops: list[TransitStop]) -> list[TransitStop]:
    from .graph import snap_to_node

    for s in stops:
        info = snap_to_node(graph, (s.lon, s.lat))
        s.node = info["node"]
        s.snap_dist_m = info["snap_dist_m"]
    return stops


def _straight_m(graph, a: int, b: int) -> float:
    return math.hypot(
        graph.nodes[a]["x"] - graph.nodes[b]["x"],
        graph.nodes[a]["y"] - graph.nodes[b]["y"],
    )


def nearest_stop(stops: list[TransitStop], node: int, graph) -> TransitStop:
    best = None
    best_d = math.inf
    for s in stops:
        if s.node is None:
            continue
        d = _straight_m(graph, node, s.node)
        if d < best_d:
            best_d = d
            best = s
    return best


def estimate_ride_min(stop_a: TransitStop, stop_b: TransitStop) -> float:
    # crude: straight distance between stops + fixed wait
    # (coords are lon/lat; ~111320 m/deg lat, ~62100 m/deg lon at 55.7)
    mlat = 111320.0
    mlon = 62100.0
    dist = math.hypot((stop_a.lon - stop_b.lon) * mlon, (stop_a.lat - stop_b.lat) * mlat)
    return DEFAULT_RIDE_WAIT_MIN + (dist / 1000.0) * DEFAULT_RIDE_MIN_PER_KM


def sample_hhmm(rng: random.Random, start: str, end: str) -> str:
    a = hhmm_to_minutes(start)
    b = hhmm_to_minutes(end)
    m = rng.randint(a, b)
    return minutes_to_hhmm(m)


def make_synthetic_user(
    rng: random.Random,
    graph,
    stops: list[TransitStop],
    user_id: str,
    cfg: HabitualConfig,
) -> SyntheticUser | None:
    nodes = list(graph.nodes)
    home = rng.choice(nodes)
    # destination at least MIN_OD_STRAIGHT_M away
    for _ in range(40):
        dest = rng.choice(nodes)
        if _straight_m(graph, home, dest) >= MIN_OD_STRAIGHT_M:
            break
    else:
        return None

    home_stop = nearest_stop(stops, home, graph)
    dest_stop = nearest_stop(stops, dest, graph)
    if home_stop is None or dest_stop is None:
        return None

    ride = estimate_ride_min(home_stop, dest_stop)
    weekday = rng.choice(["mon", "tue", "wed", "thu", "fri"])
    morning_dep = sample_hhmm(rng, *cfg.morning_range)
    evening_dep = sample_hhmm(rng, *cfg.evening_range)

    home_stop_id = f"stop_home_{user_id}"
    dest_stop_id = f"stop_dest_{user_id}"
    # landmark ids must resolve in the engine — we register user nodes separately
    routes = (
        HabitualRoute(
            id=f"{user_id}_morning",
            label="утро",
            period="morning",
            weekdays=("mon", "tue", "wed", "thu", "fri"),
            departure_time=morning_dep,
            origin="home",
            destination="dest",
            walk_legs=(
                WalkLeg("home", home_stop_id, "пешком до остановки"),
                WalkLeg(dest_stop_id, "dest", "пешком от остановки"),
            ),
            transfers=(
                Transfer("transit", home_stop_id, dest_stop_id, "поездка", ride),
            ),
        ),
        HabitualRoute(
            id=f"{user_id}_evening",
            label="вечер",
            period="evening",
            weekdays=("mon", "tue", "wed", "thu", "fri"),
            departure_time=evening_dep,
            origin="dest",
            destination="home",
            walk_legs=(
                WalkLeg("dest", dest_stop_id, "пешком до остановки"),
                WalkLeg(home_stop_id, "home", "пешком от остановки"),
            ),
            transfers=(
                Transfer("transit", dest_stop_id, home_stop_id, "поездка", ride),
            ),
        ),
    )
    return SyntheticUser(
        user_id=user_id,
        home_node=home,
        dest_node=dest,
        home_stop=home_stop,
        dest_stop=dest_stop,
        ride_min=ride,
        weekday=weekday,
        morning_dep=morning_dep,
        evening_dep=evening_dep,
        routes=routes,
    )


class UserEngine:
    """Per-user landmark node map + shared Dijkstra cache per (from,to) node pair."""

    def __init__(self, graph, cfg: HabitualConfig, service_time_min: float | None = None):
        self.graph = graph
        self.cfg = cfg
        self.service_time_min = (
            service_time_min if service_time_min is not None else cfg.service_time_min
        )
        self._board_cache: dict[tuple[int, int], Any] = {}
        self._home_dist_cache: dict[int, dict[int, float]] = {}

    def board(self, node_a: int, node_b: int):
        key = (node_a, node_b)
        if key not in self._board_cache:
            import time as time_mod

            from .detour import LegScoreBoard
            from .graph import MAX_WALK_MIN, dijkstra_with_prev, path_coordinates, reconstruct_path

            t0 = time_mod.perf_counter()
            dist_a, prev = dijkstra_with_prev(
                self.graph, node_a, max_cost=MAX_WALK_MIN
            )
            t1 = time_mod.perf_counter()
            dist_b, _ = dijkstra_with_prev(
                self.graph.reverse(copy=False), node_b, max_cost=MAX_WALK_MIN
            )
            t2 = time_mod.perf_counter()
            if node_b not in dist_a:
                raise nx.NetworkXNoPath(f"No path {node_a}→{node_b}")
            path = reconstruct_path(prev, node_a, node_b)
            self._board_cache[key] = LegScoreBoard(
                leg_from=str(node_a),
                leg_to=str(node_b),
                node_a=node_a,
                node_b=node_b,
                t_a_to_b_min=float(dist_a[node_b]),
                dist_from_a=dist_a,
                dist_to_b=dist_b,
                path_nodes=frozenset(path),
                path_coords=path_coordinates(self.graph, path),
                dijkstra_from_a_s=t1 - t0,
                dijkstra_to_b_s=t2 - t1,
            )
        return self._board_cache[key]

    def user_reachable(self, user: "SyntheticUser") -> bool:
        """True if every walk leg of every commute has a path within MAX_WALK_MIN.

        The OSM walk graph is only weakly connected (one-way footways, service
        roads), and a random home far from any stop can exceed the walking
        cutoff; such users would crash scoring with NetworkXNoPath.
        """
        node_map = _user_node_map(user)
        try:
            for route in user.routes:
                for leg in route.walk_legs:
                    self.board(node_map[leg.from_id], node_map[leg.to_id])
        except nx.NetworkXNoPath:
            return False
        return True

    def home_walk_min(self, home_node: int, point_node: int) -> float:
        if home_node not in self._home_dist_cache:
            dist, _ = dijkstra_with_prev(
                self.graph, home_node, max_cost=MAX_WALK_MIN
            )
            self._home_dist_cache[home_node] = dist
        return self._home_dist_cache[home_node].get(point_node, math.inf)

    def network_dist(self, home_node: int, point_node: int) -> float:
        return self.home_walk_min(home_node, point_node)


def _user_node_map(user: SyntheticUser) -> dict[str, int]:
    """Node map: home/dest + the two transit stops used in make_synthetic_user."""
    stop_home_id = user.routes[0].walk_legs[0].to_id  # home → stop_home_*
    stop_dest_id = user.routes[0].walk_legs[1].from_id  # stop_dest_* → dest
    return {
        "home": user.home_node,
        "dest": user.dest_node,
        stop_home_id: user.home_stop.node,
        stop_dest_id: user.dest_stop.node,
    }


def evaluate_point_on_user(
    engine: UserEngine,
    user: SyntheticUser,
    point: PickupPoint,
    weekday: str,
    day: date,
    parcel_weight_kg: float,
) -> dict | None:
    """Best commute option for one point; None if unreachable on all legs."""
    node_map = _user_node_map(user)

    options = []
    for route in user.routes:
        if weekday not in route.weekdays:
            continue
        from .detour import arrival_offset_min, commute_timeline

        events = commute_timeline(route)
        leg_dur = {}
        boards = []
        for leg in route.walk_legs:
            a = node_map[leg.from_id]
            b = node_map[leg.to_id]
            board = engine.board(a, b)
            boards.append((leg, board))
            leg_dur[(leg.from_id, leg.to_id)] = board.t_a_to_b_min
        for ev in events:
            if ev["kind"] == "walk":
                ev["leg_duration_min"] = leg_dur.get((ev["from"], ev["to"]), 0.0)

        scored = []
        for leg, board in boards:
            res = detour_for_point(board, point)
            if math.isinf(res.detour_min):
                continue
            scored.append((res, leg, board))
        if not scored:
            continue
        res, leg, board = min(scored, key=lambda x: x[0].detour_min)
        offset = arrival_offset_min(events, leg.from_id, leg.to_id, res.t_a_to_p_min)
        dep = hhmm_to_minutes(route.departure_time)
        arrive = minutes_to_hhmm(dep + offset)
        open_ok, open_reason = is_open_at(
            point, weekday, arrive, engine.service_time_min
        )
        wt_ok, wt_reason = weight_ok(point, parcel_weight_kg)
        options.append({
            "commute_id": route.id,
            "period": route.period,
            "leg_from": leg.from_id,
            "leg_to": leg.to_id,
            "detour_min": res.detour_min,
            "on_path": res.on_shortest_path,
            "arrival": arrive,
            "open": open_ok,
            "open_reason": open_reason,
            "weight_ok": wt_ok,
            "usable": open_ok and wt_ok,
            "hours_today": (
                None
                if point.hours_on(weekday)[0] is None
                else f"{point.hours_on(weekday)[0]}-{point.hours_on(weekday)[1]}"
            ),
            "close_hhmm": point.hours_on(weekday)[1],
        })
    if not options:
        return None
    best = min(options, key=lambda o: (not o["usable"], o["detour_min"]))
    best = dict(best)
    best["all_options"] = options
    return best


def round_trip_home_min(engine: UserEngine, home_node: int, point: PickupPoint) -> float:
    t = engine.home_walk_min(home_node, point.node)
    return math.inf if math.isinf(t) else 2.0 * t


def is_open_window(point: PickupPoint, weekday: str, service_min: float) -> bool:
    """True if the point has a window on `weekday` long enough for a visit."""
    start, end = point.hours_on(weekday)
    if start is None or end is None:
        return False
    return hhmm_to_minutes(end) - hhmm_to_minutes(start) >= service_min


def cost_for_point(
    engine: UserEngine,
    user: SyntheticUser,
    point: PickupPoint,
    weekday: str,
    day: date,
    parcel_weight_kg: float,
) -> dict[str, Any]:
    """Shared cost used by ALL methods.

        cost(point) = min(
            best usable detour on the user's commutes,   # stop on the way
            round trip home → point → home at a usable time,  # separate trip
        )

    Inf if neither option is available (weight, hours, or unreachable).
    """
    ev = evaluate_point_on_user(
        engine, user, point, weekday, day, parcel_weight_kg
    )

    detour_cost = math.inf
    detour_info = None
    if ev is not None and ev["usable"]:
        detour_cost = float(ev["detour_min"])
        detour_info = ev

    rt_cost = math.inf
    if point.node is not None:
        t = engine.home_walk_min(user.home_node, point.node)
        if math.isfinite(t):
            # separate trip is available if the point is open long enough that day
            # and the parcel weight fits
            if (
                is_open_window(point, weekday, engine.service_time_min)
                and parcel_weight_kg <= point.max_weight_kg
            ):
                rt_cost = 2.0 * float(t)

    cost = min(detour_cost, rt_cost)
    via = (
        "detour"
        if cost == detour_cost and math.isfinite(cost)
        else ("round_trip" if math.isfinite(cost) else "none")
    )
    return {
        "point": point,
        "eval": ev,
        "cost_min": cost,
        "via": via,
        "detour_cost_min": detour_cost,
        "round_trip_cost_min": rt_cost,
        "usable": math.isfinite(cost),
    }


def score_method(
    method: str,
    engine: UserEngine,
    user: SyntheticUser,
    points: list[PickupPoint],
    weekday: str,
    day: date,
    parcel_weight_kg: float,
) -> dict[str, Any]:
    """Score one method. ALL methods use the same cost_for_point.

    - ours              = argmin cost over all points
    - baseline_straight = cost of the straight-line closest point to home
    - baseline_network  = cost of the network-closest point to home
    """
    scored = [cost_for_point(engine, user, p, weekday, day, parcel_weight_kg) for p in points]
    if not scored:
        return {
            "method": method, "point_id": None, "extra_min": math.inf,
            "round_trip_home_min": math.inf, "detour_min": math.inf,
            "usable": False, "on_the_way": False, "via": "none",
        }

    hx, hy = engine.graph.nodes[user.home_node]["x"], engine.graph.nodes[user.home_node]["y"]
    for s in scored:
        p = s["point"]
        if p.node is not None and p.node in engine.graph.nodes:
            px, py = engine.graph.nodes[p.node]["x"], engine.graph.nodes[p.node]["y"]
            s["home_straight"] = math.hypot(px - hx, py - hy)
            s["home_network"] = engine.home_walk_min(user.home_node, p.node)
        else:
            s["home_straight"] = math.inf
            s["home_network"] = math.inf

    if method == "ours":
        pick = min(scored, key=lambda s: (not math.isfinite(s["cost_min"]), s["cost_min"]))
    elif method == "baseline_straight":
        pick = min(scored, key=lambda s: s["home_straight"])
    elif method == "baseline_network":
        pick = min(scored, key=lambda s: (not math.isfinite(s["home_network"]), s["home_network"]))
    else:
        raise ValueError(method)

    p = pick["point"]
    ev = pick["eval"]
    cost = pick["cost_min"]
    detour = pick["detour_cost_min"]
    rt = pick["round_trip_cost_min"]
    on_the_way = pick["via"] == "detour"

    return {
        "method": method,
        "point_id": p.id,
        "point_name": p.name,
        "extra_min": cost,
        "round_trip_home_min": rt,
        "detour_min": detour,
        "via": pick["via"],
        "arrival": ev["arrival"] if ev else None,
        "open": bool(ev and ev["open"]) if ev else (math.isfinite(rt)),
        "weight_ok": bool(ev and ev["weight_ok"]) if ev else (parcel_weight_kg <= p.max_weight_kg),
        "usable": math.isfinite(cost),
        "on_the_way": on_the_way,
        "home_straight_m": pick["home_straight"],
        "home_network_min": pick["home_network"],
        "hours_today": ev.get("hours_today") if ev else (
            f"{p.hours_on(weekday)[0]}-{p.hours_on(weekday)[1]}"
            if p.hours_on(weekday)[0] else None
        ),
    }


def hours_removal_stats(
    engine: UserEngine,
    user: SyntheticUser,
    points: list[PickupPoint],
    weekday: str,
    day: date,
    parcel_weight_kg: float,
    period: str,
) -> dict[str, int]:
    """Count options removed by opening hours for one period."""
    total = 0
    removed = 0
    for p in points:
        ev = evaluate_point_on_user(engine, user, p, weekday, day, parcel_weight_kg)
        if not ev:
            continue
        for opt in ev.get("all_options", []):
            if opt["period"] != period:
                continue
            if not opt["weight_ok"]:
                continue
            total += 1
            if not opt["open"]:
                removed += 1
    return {"total": total, "removed_hours": removed}


def days_until_pickup(
    engine: UserEngine,
    user: SyntheticUser,
    points: list[PickupPoint],
    method: str,
    arrival_weekday: str,
    parcel_weight_kg: float,
    storage_days: int = 7,
) -> int | None:
    """Days from parcel arrival until the first usable pickup with this method.

    0 = same day. None = no opportunity within `storage_days`.
    Arrival weekday is `arrival_weekday`; each next day advances the calendar.
    Commutes exist only Mon–Fri (as in the config), but a separate round trip
    is possible any day the point is open.
    """
    wd_index = WD_EN.index(arrival_weekday)
    for d in range(storage_days + 1):
        wd = WD_EN[(wd_index + d) % 7]
        # reference date aligned to weekday `wd`
        day = date(2026, 10, 12) + timedelta(days=WD_EN.index(wd))
        result = score_method(
            method, engine, user, points, wd, day, parcel_weight_kg
        )
        if result["usable"]:
            return d
    return None


def days_until_on_the_way(
    engine: UserEngine,
    user: SyntheticUser,
    point: PickupPoint | None,
    arrival_weekday: str,
    parcel_weight_kg: float,
    max_detour_min: float = 5.0,
    storage_days: int = 7,
) -> int | None:
    """Days until the chosen point can be picked up ON THE WAY (detour ≤ max).

    Unlike `days_until_pickup`, a separate round trip does not count, so this
    measures the delay that "pick up on the way" really adds (Ma et al., 2024).
    0 = same day, None = no such commute within `storage_days`.
    """
    if point is None:
        return None
    wd_index = WD_EN.index(arrival_weekday)
    for d in range(storage_days + 1):
        wd = WD_EN[(wd_index + d) % 7]
        day = date(2026, 10, 12) + timedelta(days=WD_EN.index(wd))
        ev = evaluate_point_on_user(engine, user, point, wd, day, parcel_weight_kg)
        if ev is not None and ev["usable"] and ev["detour_min"] <= max_detour_min:
            return d
    return None
