"""Days from parcel arrival until the first usable pickup opportunity.

Metric (Ma et al., 2024 critique): "on the way" pickup may delay collection.
For a chosen pickup point and a user's habitual commutes, count calendar days
from a random arrival weekday until the first day on which some commute offers
a usable pickup (open at arrival + weight OK + reachable leg). 0 = same day.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

WEEKDAYS: tuple[str, ...] = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
STORAGE_DAYS_DEFAULT = 7


@dataclass
class DelayResult:
    delay_days: int | None  # None = no opportunity within max_days
    arrival_weekday: str
    within_1_day: bool
    within_storage: bool
    first_opportunity_weekday: str | None
    storage_days: int

    @property
    def censored(self) -> bool:
        return self.delay_days is None


def weekday_after(start_weekday: str, days: int) -> str:
    if start_weekday not in WEEKDAYS:
        raise ValueError(f"Unknown weekday {start_weekday!r}")
    i = WEEKDAYS.index(start_weekday)
    return WEEKDAYS[(i + days) % 7]


def days_to_first_usable(
    arrival_weekday: str,
    usable_on: Callable[[str], bool],
    *,
    storage_days: int = STORAGE_DAYS_DEFAULT,
    max_days: int | None = None,
) -> DelayResult:
    """Scan day-by-day from arrival weekday until `usable_on(weekday)` is true.

    `usable_on` must answer whether that weekday's commutes allow a usable
    pickup of the already-chosen point. Weekends are still calendar days even
    if the user has no commutes (usable_on should return False then).
    """
    if arrival_weekday not in WEEKDAYS:
        raise ValueError(f"Unknown weekday {arrival_weekday!r}")
    limit = max_days if max_days is not None else max(storage_days, 7)

    for k in range(limit + 1):
        wd = weekday_after(arrival_weekday, k)
        if usable_on(wd):
            return DelayResult(
                delay_days=k,
                arrival_weekday=arrival_weekday,
                within_1_day=k <= 1,
                within_storage=k <= storage_days,
                first_opportunity_weekday=wd,
                storage_days=storage_days,
            )

    return DelayResult(
        delay_days=None,
        arrival_weekday=arrival_weekday,
        within_1_day=False,
        within_storage=False,
        first_opportunity_weekday=None,
        storage_days=storage_days,
    )


def summarize_delays(
    results: Sequence[DelayResult | None],
    *,
    storage_days: int = STORAGE_DAYS_DEFAULT,
) -> dict:
    """Mean/median delay and share ≤1 day / ≤ storage window.

    Censored observations (no opportunity within the horizon) are excluded
    from mean/median but counted in the share denominators (as misses).
    """
    n = len(results)
    delays = [r.delay_days for r in results if r is not None and r.delay_days is not None]
    within_1 = [r for r in results if r is not None and r.within_1_day]
    within_s = [r for r in results if r is not None and r.within_storage]

    mean_d = (sum(delays) / len(delays)) if delays else None
    if delays:
        s = sorted(delays)
        mid = len(s) // 2
        median_d = s[mid] if len(s) % 2 else 0.5 * (s[mid - 1] + s[mid])
    else:
        median_d = None

    return {
        "n": n,
        "n_observed": len(delays),
        "n_censored": n - len(delays),
        "mean_delay_days": mean_d,
        "median_delay_days": median_d,
        "share_within_1_day": (len(within_1) / n) if n else 0.0,
        "share_within_storage": (len(within_s) / n) if n else 0.0,
        "storage_days": storage_days,
    }
