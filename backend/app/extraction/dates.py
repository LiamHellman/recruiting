"""Parse CV dates as written ("Mar 2019", "2019", "03/2019", "present") into the last day
of the period they name. End-of-period is the reading most favourable to the candidate,
so a vague date never makes evidence look older than it is. Pure; no I/O."""

import calendar
import re
from datetime import date

# English names, spelled out: calendar.month_name follows the process locale.
_MONTH_NAMES = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MONTHS = {name: n for n, full in enumerate(_MONTH_NAMES, 1) for name in (full, full[:3])}
_MONTHS["sept"] = 9
_PRESENT = {"present", "current", "currently", "now", "today", "ongoing", "to date"}
_SEASONS = {"spring": 6, "summer": 8, "fall": 12, "autumn": 12, "winter": 3}


def _month_end(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def period_end(written: str | None, *, today: date) -> date | None:
    """The last day of the period `written` names; `today` for "present". None if the text
    is not a date we recognise. Never later than `today`."""
    if written is None:
        return None
    text = written.strip().lower().rstrip(".")
    if not text:
        return None
    if text in _PRESENT:
        return today
    result: date | None = None
    if m := re.fullmatch(r"((?:19|20)\d\d)", text):
        result = date(int(m[1]), 12, 31)
    elif m := re.fullmatch(r"([a-z]+)\.?,?\s+((?:19|20)\d\d)", text):
        name, year = m[1], int(m[2])
        if name in _MONTHS:
            result = _month_end(year, _MONTHS[name])
        elif name in _SEASONS:
            result = _month_end(year, _SEASONS[name])
    elif m := re.fullmatch(r"(\d{1,2})\s*[/.-]\s*((?:19|20)\d\d)", text):
        if 1 <= int(m[1]) <= 12:
            result = _month_end(int(m[2]), int(m[1]))
    elif m := re.fullmatch(r"((?:19|20)\d\d)\s*[/.-]\s*(\d{1,2})", text):
        if 1 <= int(m[2]) <= 12:
            result = _month_end(int(m[1]), int(m[2]))
    elif m := re.fullmatch(r"q([1-4])\s+((?:19|20)\d\d)", text):
        result = _month_end(int(m[2]), int(m[1]) * 3)
    if result is None:
        return None
    return min(result, today)
