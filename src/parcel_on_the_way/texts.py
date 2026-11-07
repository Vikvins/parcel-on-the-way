"""Russian user-facing text helpers for notifications and explanations."""

from __future__ import annotations

from datetime import date

WD_RU = {
    "mon": "понедельник",
    "tue": "вторник",
    "wed": "среда",
    "thu": "четверг",
    "fri": "пятница",
    "sat": "суббота",
    "sun": "воскресенье",
}
WD_RU_GEN = {  # genitive after "в"
    "mon": "понедельник",
    "tue": "вторник",
    "wed": "среду",
    "thu": "четверг",
    "fri": "пятницу",
    "sat": "субботу",
    "sun": "воскресенье",
}
MONTHS_RU = {
    1: "января", 2: "февраля", 3: "марта", 4: "апреля",
    5: "мая", 6: "июня", 7: "июля", 8: "августа",
    9: "сентября", 10: "октября", 11: "ноября", 12: "декабря",
}


def format_date_ru(day: date, weekday: str | None = None) -> str:
    """'в понедельник, 19 октября'."""
    wd = weekday or ["mon", "tue", "wed", "thu", "fri", "sat", "sun"][day.weekday()]
    return f"в {WD_RU_GEN[wd]}, {day.day} {MONTHS_RU[day.month]}"


def format_date_short_ru(day: date) -> str:
    """'19 октября'."""
    return f"{day.day} {MONTHS_RU[day.month]}"


def format_weekday_ru(weekday: str, with_prefix: bool = True) -> str:
    name = WD_RU.get(weekday, weekday)
    return f"в {name}" if with_prefix else name


def format_detour_ru(detour_min: float) -> str:
    if detour_min < 0.5:
        return "прямо по пути"
    return f"+{detour_min:.0f} мин"


def format_close_ru(close_hhmm: str | None, hours_today: str | None = None) -> str:
    """'круглосуточно' / 'открыт до 21:00' / 'закрыт'."""
    if hours_today in ("00:00-23:59", "00:00-24:00") or close_hhmm in ("23:59", "24:00"):
        return "круглосуточно"
    if close_hhmm is None:
        return "сегодня закрыт"
    return f"открыт до {close_hhmm}"


def format_period_ru(period: str) -> str:
    return "утром по дороге в вуз" if period == "morning" else "вечером по дороге из вуза домой"


def format_notification_detour(
    period: str,
    detour_min: float,
    close_hhmm: str | None,
    hours_today: str | None,
) -> str:
    """Human notification, no technical IDs or timestamps of arrival.

    'Можно забрать утром по дороге в вуз, +1 мин. Пункт открыт до 18:00.'
    'Можно забрать вечером по дороге из вуза домой, прямо по пути. Пункт круглосуточный.'
    """
    detour_s = format_detour_ru(detour_min)
    if detour_min >= 0.5:
        detour_s = f"крюк {detour_s}"
    head = f"Можно забрать {format_period_ru(period)}, {detour_s}."
    if hours_today in ("00:00-23:59", "00:00-24:00") or close_hhmm in ("23:59", "24:00"):
        tail = "Пункт работает круглосуточно."
    elif close_hhmm:
        tail = f"Пункт открыт до {close_hhmm}."
    else:
        tail = "Пункт сегодня закрыт."
    return f"{head} {tail}"


def format_weight_rejection(point_name: str, parcel_kg: float, max_kg: float) -> str:
    return (
        f"Пункт «{point_name}» не принимает посылки весом {parcel_kg:g} кг "
        f"(лимит {max_kg:g} кг). Другой день не поможет — нужен другой пункт "
        f"или доставка курьером."
    )
