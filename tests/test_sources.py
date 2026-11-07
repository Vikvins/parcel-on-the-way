"""Parsers for real CDEK exports (no graph needed)."""

import json

import pytest

from parcel_on_the_way.hours import is_open_at
from parcel_on_the_way.points import load_any_points, parse_hours_cell
from parcel_on_the_way.sources import (
    ALL_DAY,
    CLOSED,
    load_cdek_api_json,
    load_overpass_json,
    normalize_range,
    parse_osm_opening_hours,
    parse_ru_work_time,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("10:00/21:00", ("10:00", "21:00")),
        ("9.00–21.00", ("09:00", "21:00")),
        ("00:00/00:00", ALL_DAY),
        ("круглосуточно", ALL_DAY),
        ("10:00-00:00", ("10:00", "24:00")),
        ("", CLOSED),
        ("выходной", CLOSED),
    ],
)
def test_normalize_range(text, expected):
    assert normalize_range(text) == expected


def test_hours_cell_accepts_24h_and_dash_variants():
    assert parse_hours_cell("00:00-24:00") == ALL_DAY
    assert parse_hours_cell("10:00–21:00") == ("10:00", "21:00")
    assert parse_hours_cell("closed") == CLOSED
    with pytest.raises(ValueError):
        parse_hours_cell("с утра")


def test_ru_work_time():
    h = parse_ru_work_time("Пн-Пт 10:00-21:00, Сб-Вс 10:00-18:00")
    assert h["mon"] == ("10:00", "21:00") and h["fri"] == ("10:00", "21:00")
    assert h["sat"] == ("10:00", "18:00") and h["sun"] == ("10:00", "18:00")
    h = parse_ru_work_time("Пн-Сб 09:00-20:00")
    assert h["sat"] == ("09:00", "20:00") and h["sun"] == CLOSED


def test_osm_opening_hours():
    h = parse_osm_opening_hours("Mo-Fr 10:00-21:00; Sa,Su 11:00-18:00; PH off")
    assert h["wed"] == ("10:00", "21:00")
    assert h["sun"] == ("11:00", "18:00")
    assert parse_osm_opening_hours("24/7")["sat"] == ALL_DAY
    with pytest.raises(ValueError):
        parse_osm_opening_hours("Jan-Mar Mo 10:00-12:00")


def test_all_day_point_open_late():
    from parcel_on_the_way.points import PickupPoint

    p = PickupPoint("x", "x", 0, 0, "postamat", 20, {d: ALL_DAY for d in
                    ("mon", "tue", "wed", "thu", "fri", "sat", "sun")})
    assert is_open_at(p, "wed", "23:50", 4)[0]
    assert is_open_at(p, "wed", "06:10", 4)[0]


API_SAMPLE = [
    {
        "code": "MSK123",
        "type": "PVZ",
        "is_handout": True,
        "weight_max": 30,
        "location": {"latitude": 55.70, "longitude": 37.58, "address": "ул. Вавилова, 17А"},
        "work_time_list": [{"day": d, "time": "10:00/21:00"} for d in range(1, 6)]
        + [{"day": 6, "time": "10:00/18:00"}],
    },
    {
        "code": "MSK999",
        "type": "POSTAMAT",
        "location": {"latitude": 55.71, "longitude": 37.59, "address": "Ленинский пр., 30"},
        "work_time": "Пн-Вс 00:00-23:59",
    },
    {"code": "MSK000", "type": "PVZ", "is_handout": False,
     "location": {"latitude": 55.7, "longitude": 37.6}},
]


def test_cdek_api_json(tmp_path):
    f = tmp_path / "api.json"
    f.write_text(json.dumps({"items": API_SAMPLE}, ensure_ascii=False), encoding="utf-8")
    pts = load_any_points(f)
    assert [p.id for p in pts] == ["MSK123", "MSK999"]  # reception-only dropped
    a, b = pts
    assert a.max_weight_kg == 30 and a.hours["sat"] == ("10:00", "18:00")
    assert a.hours["sun"] == CLOSED
    assert b.point_type == "postamat" and b.hours["sun"] == ALL_DAY
    assert load_cdek_api_json(f)[0].name.startswith("СДЭК")


def test_overpass_json(tmp_path):
    f = tmp_path / "osm.json"
    f.write_text(json.dumps({"elements": [
        {"type": "node", "id": 1, "lat": 55.7, "lon": 37.58,
         "tags": {"brand": "СДЭК", "opening_hours": "Mo-Su 09:00-21:00"}},
        {"type": "way", "id": 2, "center": {"lat": 55.71, "lon": 37.59},
         "tags": {"name": "CDEK", "amenity": "parcel_locker", "opening_hours": "weird"}},
        {"type": "node", "id": 3, "lat": 55.7, "lon": 37.6, "tags": {"name": "Почта"}},
    ]}), encoding="utf-8")
    pts = load_overpass_json(f)
    assert [p.id for p in pts] == ["OSM-node1", "OSM-way2"]
    assert pts[0].hours["sun"] == ("09:00", "21:00")
    assert pts[1].point_type == "postamat" and "unknown" in pts[1].note


def test_collected_csv_format(tmp_path):
    """Format of data/cdek_real/cdek_moscow_points.csv (no source_url/checked_date)."""
    f = tmp_path / "cdek_moscow_points.csv"
    f.write_text(
        "id,source,code,address,type,lat,lon,hours_mon,hours_tue,hours_wed,hours_thu,"
        "hours_fri,hours_sat,hours_sun,max_weight_kg,opening_hours_raw\n"
        "MSK1,cdek_api,MSK1,\"ул. Вавилова, 17А\",ПВЗ,55.70,37.58,10:00-21:00,10:00-21:00,"
        "10:00-21:00,10:00-21:00,10:00-21:00,,,,\n"
        "MSK2,osm,,Ленинский 30,постамат,55.71,37.59,00:00-24:00,00:00-24:00,00:00-24:00,"
        "00:00-24:00,00:00-24:00,00:00-24:00,00:00-24:00,20,24/7\n",
        encoding="utf-8",
    )
    pts = load_any_points(f)
    assert pts[0].hours["sat"] == CLOSED and pts[0].max_weight_kg == 31.0
    assert pts[1].hours["sun"] == ALL_DAY and pts[1].max_weight_kg == 20


def test_rows_without_coords_are_skipped(tmp_path):
    f = tmp_path / "pts.csv"
    head = ("id,address,type,lat,lon,hours_mon,hours_tue,hours_wed,hours_thu,"
            "hours_fri,hours_sat,hours_sun,max_weight_kg\n")
    row = ",10:00-21:00,10:00-21:00,10:00-21:00,10:00-21:00,10:00-21:00,,,30\n"
    f.write_text(head + "A,x,ПВЗ,55.7,37.6" + row + "B,y,ПВЗ,," + row, encoding="utf-8")
    with pytest.warns(UserWarning, match="1 point"):
        pts = load_any_points(f)
    assert [p.id for p in pts] == ["A"]
    f.write_text(head + "B,y,ПВЗ,," + row, encoding="utf-8")
    with pytest.raises(ValueError):
        load_any_points(f)
