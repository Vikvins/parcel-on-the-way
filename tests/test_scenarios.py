"""Unit tests for hours filter and scenario API edge cases."""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from parcel_on_the_way.hours import is_open_at, weight_ok  # noqa: E402
from parcel_on_the_way.points import PickupPoint, load_pickup_points  # noqa: E402
from parcel_on_the_way.routes_config import load_habitual_config  # noqa: E402


def _point(hours_wed=("09:00", "18:00"), max_kg=15.0, **kwargs) -> PickupPoint:
    hours = {d: (None, None) for d in
             ("mon", "tue", "wed", "thu", "fri", "sat", "sun")}
    hours["wed"] = hours_wed
    return PickupPoint(
        id=kwargs.get("id", "P1"),
        name=kwargs.get("name", "Test"),
        lat=55.7, lon=37.6,
        point_type="pvz",
        max_weight_kg=max_kg,
        hours=hours,
    )


# ---- service-time filter ----------------------------------------------------

def test_open_when_arrival_and_service_fit():
    p = _point(("09:00", "18:00"))
    ok, reason = is_open_at(p, "wed", "17:50", 4.0)
    assert ok, reason


def test_cut_only_because_of_service_time():
    """arrival 17:58 < close 18:00, but +4 min service spills past close."""
    p = _point(("09:00", "18:00"))
    ok, reason = is_open_at(p, "wed", "17:58", 4.0)
    assert not ok
    assert "after close" in reason
    # same arrival with shorter service still works
    ok2, _ = is_open_at(p, "wed", "17:58", 2.0)
    assert ok2


def test_cut_when_arrive_before_open_no_waiting():
    p = _point(("10:00", "21:00"))
    ok, reason = is_open_at(p, "wed", "09:50", 4.0)
    assert not ok
    assert "before open" in reason


def test_closed_on_weekday_not_friday():
    p = _point(("09:00", "18:00"))
    p.hours["thu"] = (None, None)
    ok, reason = is_open_at(p, "thu", "12:00", 4.0)
    assert not ok
    assert "closed on thu" in reason
    # same time on wed is fine
    ok2, _ = is_open_at(p, "wed", "12:00", 4.0)
    assert ok2


def test_weight_limit():
    p = _point(max_kg=5.0)
    assert weight_ok(p, 5.0)[0] is True
    assert weight_ok(p, 5.1)[0] is False


# ---- config / CSV sanity ----------------------------------------------------

def test_yaml_has_ride_minutes_and_ranges():
    cfg = load_habitual_config()
    assert cfg.service_time_min == 4.0
    assert cfg.morning_range == ("08:00", "10:30")
    assert cfg.evening_range == ("16:00", "21:00")
    assert len(cfg.routes) == 4
    for r in cfg.routes:
        assert r.total_ride_min > 0, r.id
        for t in r.transfers:
            assert t.ride_min > 0, (r.id, t)


def test_csv_ten_points_and_short_hours():
    points = load_pickup_points()
    assert len(points) == 12
    short = [p for p in points if p.hours_on("wed")[1] in ("17:00", "18:00")]
    assert len(short) >= 2, "need short-hours points for evening filter"
    assert any(p.point_type == "postamat" for p in points)
    assert any(p.max_weight_kg <= 5.0 for p in points)


# ---- scenario API (needs graph; skip if not built) --------------------------

@pytest.fixture(scope="module")
def engine_bundle():
    graphml = ROOT / "data" / "processed" / "moscow_walk.graphml"
    if not graphml.exists():
        pytest.skip("graph not built")
    from parcel_on_the_way.graph import load_graph
    from parcel_on_the_way.points import snap_landmarks, snap_points
    from parcel_on_the_way.scenarios import DetourEngine

    cfg = load_habitual_config()
    g = load_graph()
    points = load_pickup_points()
    snap_points(g, points)
    landmarks = {k: info["node"] for k, info in snap_landmarks(g).items()}
    engine = DetourEngine(g, landmarks, cfg)
    return engine, points, cfg


def test_scenario1_top3_distinct_reasons(engine_bundle):
    from parcel_on_the_way.scenarios import scenario1_pick_point

    engine, points, _ = engine_bundle
    s1 = scenario1_pick_point(engine, points, day="2026-10-14", parcel_weight_kg=2.0)
    assert s1["n_candidates"] >= 1
    results = s1["results"]
    assert 1 <= len(results) <= 3
    ids = [r["point_id"] for r in results]
    assert len(ids) == len(set(ids)), "top-3 points must be distinct"
    reasons = [r["reason"] for r in results]
    assert reasons[0] == "detour"
    assert all(r["usable"] for r in results)
    # every candidate is usable on at least one commute
    assert all(c["usable"] for c in s1["candidates"])


def test_scenario1_does_not_drop_morning_closed_point(engine_bundle):
    """PVZ-TEST-07 opens 09:00 — unusable early morning, but evening/morning
    later commute may still qualify it. It must appear as candidate if ANY
    commute works."""
    from parcel_on_the_way.scenarios import scenario1_pick_point

    engine, points, _ = engine_bundle
    s1 = scenario1_pick_point(engine, points, day="2026-10-14", parcel_weight_kg=2.0)
    cand_ids = {c["point_id"] for c in s1["candidates"]}
    # PVZ-TEST-07 is open 09:00-18:00; morning arrival ~09:01 works
    assert "PVZ-TEST-07" in cand_ids


def test_scenario2_fallback_to_morning_when_evening_closes(engine_bundle):
    from parcel_on_the_way.scenarios import scenario2_pickup_on_way

    engine, points, _ = engine_bundle
    s2 = scenario2_pickup_on_way(
        engine, points, point_id="PVZ-TEST-07", day="2026-10-14", parcel_weight_kg=2.0
    )
    assert s2["ok"] is True
    chosen = s2["chosen"]
    assert chosen is not None, "morning option should still work on Wednesday"
    assert chosen["usable"] is True
    assert chosen["arrival"] < "18:00"
    assert "notification_ru" in s2 and len(s2["notification_ru"]) > 10


def test_scenario2_weight_rejects_and_may_fallback_or_fail(engine_bundle):
    from parcel_on_the_way.scenarios import scenario2_pickup_on_way

    engine, points, _ = engine_bundle
    # PVZ-TEST-10 max 5 kg; 8 kg must never be usable
    s2 = scenario2_pickup_on_way(
        engine, points, point_id="PVZ-TEST-10", day="2026-10-14", parcel_weight_kg=8.0
    )
    assert s2["ok"] is False or s2["chosen"] is None
    if s2["ok"] is False:
        assert "кг" in s2["notification_ru"] or "кг" in s2.get("reason_ru", "")


def test_detour_nonnegative_and_on_path_zero(engine_bundle):
    from parcel_on_the_way.detour import detour_for_point, score_leg
    from parcel_on_the_way.scenarios import DetourEngine

    engine, points, cfg = engine_bundle
    route = next(r for r in cfg.routes if r.id == "morning_tram_metro")
    leg = route.walk_legs[0]
    board = engine.board(route.id, leg.from_id, leg.to_id)
    for p in points:
        res = detour_for_point(board, p)
        if res.detour_min != float("inf"):
            assert res.detour_min >= -1e-9
            if res.on_shortest_path:
                assert res.detour_min == pytest.approx(0.0, abs=1e-3)


# ---- Russian user-facing texts ----------------------------------------------

def test_detour_phrasing():
    from parcel_on_the_way.texts import format_detour_ru

    assert format_detour_ru(0.0) == "прямо по пути"
    assert format_detour_ru(0.4) == "прямо по пути"
    assert format_detour_ru(1.2) == "+1 мин"
    assert format_detour_ru(2.6) == "+3 мин"


def test_round_the_clock_phrasing():
    from parcel_on_the_way.texts import format_close_ru

    assert format_close_ru("23:59", "00:00-23:59") == "круглосуточно"
    assert format_close_ru("21:00", "10:00-21:00") == "открыт до 21:00"
    assert format_close_ru(None, None) == "сегодня закрыт"


def test_date_in_russian():
    from datetime import date

    from parcel_on_the_way.texts import format_date_ru

    s = format_date_ru(date(2026, 10, 19), "mon")
    assert s == "в понедельник, 19 октября"
    assert "mon" not in s and "Mon" not in s


def test_notification_is_human():
    from parcel_on_the_way.texts import format_notification_detour

    n = format_notification_detour("morning", 1.18, "18:00", "09:00-18:00")
    assert n.startswith("Можно забрать утром по дороге в вуз, крюк +1 мин.")
    assert "18:00" in n
    assert "приход" not in n and "commute" not in n and "09:0" not in n

    n2 = format_notification_detour("evening", 0.0, "23:59", "00:00-23:59")
    assert "прямо по пути" in n2 and "круглосуточно" in n2


def test_weight_rejection_is_direct():
    from parcel_on_the_way.texts import format_weight_rejection

    s = format_weight_rejection("Тест ПВЗ «Ударница»", 8.0, 5.0)
    assert "8" in s and "5" in s
    assert "Другой день не поможет" in s
