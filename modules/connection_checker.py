"""
connection_checker.py — Assesses whether a passenger can catch connecting trains.

For each interchange in a route, compares:
  predicted_arrival = scheduled_arrival + predicted_delay
  window           = next_train_departure - predicted_arrival

Status:
  ✅  SAFE         window > 30 min
  ⚠️  RISKY        15 < window <= 30 min
  ❌  NOT POSSIBLE  window <= 15 min
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List

from modules.route_planner import Route, Leg
from modules.delay_predictor import DelayPredictor


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class ConnectionStatus:
    interchange_station: str
    leg1_train:          int
    leg2_train:          str
    scheduled_arrival:   str        # HH:MM of leg1 at interchange
    predicted_delay:     float      # minutes
    predicted_arrival:   str        # HH:MM adjusted
    next_departure:      str        # HH:MM of leg2 from interchange
    window_min:          float      # minutes of breathing room
    status:              str        # SAFE / RISKY / NOT POSSIBLE
    emoji:               str
    confidence:          float


SAFE_THRESHOLD  = 30   # minutes
RISKY_THRESHOLD = 15   # minutes


def _add_minutes(hhmm: str, mins: float) -> str:
    """Return HH:MM string after adding `mins` minutes."""
    try:
        h, m = map(int, hhmm.split(":"))
    except Exception:
        return hhmm
    total = h * 60 + m + int(mins)
    return f"{(total // 60) % 24:02d}:{total % 60:02d}"


# ─── CHECKER ──────────────────────────────────────────────────────────────────

class ConnectionChecker:

    def __init__(self, predictor: DelayPredictor, rg=None):
        self.predictor = predictor
        self.rg = rg   # RailwayGraph (optional, for zone lookup)

    def check_route(self, route: Route, day_of_week: int, month: int) -> List[ConnectionStatus]:
        """Return one ConnectionStatus per interchange leg pair."""
        results: List[ConnectionStatus] = []

        for i in range(len(route.legs) - 1):
            leg1: Leg = route.legs[i]
            leg2: Leg = route.legs[i + 1]

            interchange = leg1.to_station
            zone = "NR"
            if self.rg:
                info = self.rg.station_info(interchange)
                if info:
                    zone = info.get("zone", "NR")

            sched_hour = leg1.arr_abs % (24 * 60) // 60

            delay_min, confidence = self.predictor.predict(
                train_id=leg1.train_id,
                station_code=interchange,
                scheduled_hour=sched_hour,
                day_of_week=day_of_week,
                month=month,
                zone=zone,
            )

            predicted_arr_abs = leg1.arr_abs + delay_min
            # Next departure might be next day
            next_dep_abs = leg2.dep_abs
            if next_dep_abs < leg1.arr_abs:
                next_dep_abs += 24 * 60

            window = next_dep_abs - predicted_arr_abs

            if window > SAFE_THRESHOLD:
                status, emoji = "SAFE", "[OK]"
            elif window > RISKY_THRESHOLD:
                status, emoji = "RISKY", "[!!]"
            else:
                status, emoji = "NOT POSSIBLE", "[NO]"

            results.append(ConnectionStatus(
                interchange_station=leg1.to_name,
                leg1_train=leg1.train_id,
                leg2_train=str(leg2.train_id),
                scheduled_arrival=leg1.arr_time,
                predicted_delay=delay_min,
                predicted_arrival=_add_minutes(leg1.arr_time, delay_min),
                next_departure=leg2.dep_time,
                window_min=round(window, 1),
                status=status,
                emoji=emoji,
                confidence=confidence,
            ))

        return results
