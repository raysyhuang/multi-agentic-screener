"""Trading day helpers — weekday-aware, no external holiday deps."""

from datetime import date, timedelta

# NYSE full-day closures for 2020-2027 (static set, extend as needed). Includes
# the unscheduled 2025-01-09 national day of mourning. Extend before 2028.
_NYSE_HOLIDAYS: set[date] = {
    # 2020
    date(2020, 1, 1), date(2020, 1, 20), date(2020, 2, 17),
    date(2020, 4, 10), date(2020, 5, 25), date(2020, 7, 3),
    date(2020, 9, 7), date(2020, 11, 26), date(2020, 12, 25),
    # 2021
    date(2021, 1, 1), date(2021, 1, 18), date(2021, 2, 15),
    date(2021, 4, 2), date(2021, 5, 31), date(2021, 7, 5),
    date(2021, 9, 6), date(2021, 11, 25), date(2021, 12, 24),
    # 2022
    date(2022, 1, 17), date(2022, 2, 21), date(2022, 4, 15),
    date(2022, 5, 30), date(2022, 6, 20), date(2022, 7, 4),
    date(2022, 9, 5), date(2022, 11, 24), date(2022, 12, 26),
    # 2023
    date(2023, 1, 2), date(2023, 1, 16), date(2023, 2, 20),
    date(2023, 4, 7), date(2023, 5, 29), date(2023, 6, 19),
    date(2023, 7, 4), date(2023, 9, 4), date(2023, 11, 23),
    date(2023, 12, 25),
    # 2024
    date(2024, 1, 1), date(2024, 1, 15), date(2024, 2, 19),
    date(2024, 3, 29), date(2024, 5, 27), date(2024, 6, 19),
    date(2024, 7, 4), date(2024, 9, 2), date(2024, 11, 28),
    date(2024, 12, 25),
    # 2025
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19),
    date(2025, 7, 4), date(2025, 9, 1), date(2025, 11, 27),
    date(2025, 12, 25),
    # 2026
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
    date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
    date(2026, 7, 3), date(2026, 9, 7), date(2026, 11, 26),
    date(2026, 12, 25),
    # 2027
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15),
    date(2027, 3, 26), date(2027, 5, 31), date(2027, 6, 18),
    date(2027, 7, 5), date(2027, 9, 6), date(2027, 11, 25),
    date(2027, 12, 24),
}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _NYSE_HOLIDAYS


def next_trading_day(d: date) -> date:
    nxt = d + timedelta(days=1)
    while not is_trading_day(nxt):
        nxt += timedelta(days=1)
    return nxt


def trading_days_between(start: date, end: date) -> int:
    """Count trading days between start (exclusive) and end (inclusive)."""
    if start >= end:
        return 0
    count = 0
    current = start + timedelta(days=1)
    while current <= end:
        if is_trading_day(current):
            count += 1
        current += timedelta(days=1)
    return count


def previous_trading_day(d: date) -> date:
    prev = d - timedelta(days=1)
    while not is_trading_day(prev):
        prev -= timedelta(days=1)
    return prev


def trading_sessions(start: date, end: date) -> list[date]:
    """Every XNYS session in [start, end], inclusive, in order."""
    out: list[date] = []
    current = start
    while current <= end:
        if is_trading_day(current):
            out.append(current)
        current += timedelta(days=1)
    return out
