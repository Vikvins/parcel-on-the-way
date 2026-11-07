"""Hours are checked on every walking leg before the best leg is chosen;
overnight windows close correctly in the early-morning part."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.detour import LegScoreBoard  # noqa: E402
from parcel_on_the_way.hours import is_open_at  # noqa: E402
from parcel_on_the_way.points import PickupPoint  # noqa: E402
from parcel_on_the_way.routes_config import HabitualRoute, Transfer, WalkLeg  # noqa: E402
from parcel_on_the_way.scenarios import _evaluate_point_on_commute  # noqa: E402

DAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _point(wed_hours, node=7) -> PickupPoint:
    hours = {d: (None, None) for d in DAYS}
    hours["wed"] = wed_hours
    p = PickupPoint(id="P", name="Test", lat=55.7, lon=37.6, point_type="pvz",
                    max_weight_kg=30.0, hours=hours)
    p.node = node
    return p


def _board(a, b, t_ab, t_ap, t_pb, node=7) -> LegScoreBoard:
    return LegScoreBoard(leg_from=a, leg_to=b, node_a=0, node_b=1, t_a_to_b_min=t_ab,
                         dist_from_a={node: t_ap, 1: t_ab}, dist_to_b={node: t_pb, 0: t_ab},
                         path_nodes=frozenset(), path_coords=[],
                         dijkstra_from_a_s=0.0, dijkstra_to_b_s=0.0)


class _Engine:
    """Two walking legs joined by a 40-minute metro ride."""

    def __init__(self):
        self.cfg = SimpleNamespace(service_time_min=4.0)
        self._b = {
            ("home", "metro_a"): _board("home", "metro_a", 10, 1, 9),   # detour 0
            ("metro_b", "uni"): _board("metro_b", "uni", 10, 5, 8),     # detour 3
        }

    def board(self, route_id, a, b):
        return self._b[(a, b)]

    def home_walk_min(self, point):
        return 1.0


ROUTE = HabitualRoute(
    id="r", label="test", period="morning", weekdays=("wed",), departure_time="08:00",
    origin="home", destination="uni",
    walk_legs=(WalkLeg("home", "metro_a", "w1"), WalkLeg("metro_b", "uni", "w2")),
    transfers=(Transfer("metro", "metro_a", "metro_b", "ride", 40.0),),
)


def test_open_leg_wins_over_smaller_detour_on_closed_leg():
    # leg 1: arrival 08:01, before opening at 08:30; leg 2: arrival 08:55, open.
    p = _point(("08:30", "20:00"))
    ev = _evaluate_point_on_commute(_Engine(), ROUTE, p, "wed", date(2026, 10, 14), 2.0)
    assert ev["usable"], ev["open_reason"]
    assert ev["leg_from"] == "metro_b"
    assert ev["detour_min"] == 3.0


def test_smallest_detour_when_all_legs_open():
    p = _point(("07:00", "20:00"))
    ev = _evaluate_point_on_commute(_Engine(), ROUTE, p, "wed", date(2026, 10, 14), 2.0)
    assert ev["usable"] and ev["leg_from"] == "home" and ev["detour_min"] == 0.0


def test_reports_not_usable_when_no_leg_open():
    p = _point(("10:00", "20:00"))
    ev = _evaluate_point_on_commute(_Engine(), ROUTE, p, "wed", date(2026, 10, 14), 2.0)
    assert not ev["usable"] and ev["detour_min"] == 0.0


def test_overnight_window_handover_must_end_by_close():
    p = _point(("22:00", "06:00"))
    ok, _ = is_open_at(p, "wed", "05:59", 4.0)
    assert not ok
    ok, _ = is_open_at(p, "wed", "05:50", 4.0)
    assert ok
    ok, _ = is_open_at(p, "wed", "23:00", 4.0)
    assert ok
