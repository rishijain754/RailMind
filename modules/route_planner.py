"""
route_planner.py  — Finds up to k train routes between two stations.

Strategy:
  1. Check for direct trains (one train from source to destination).
  2. Use NetworkX simple_paths (Yen-style) up to max depth 2 hops
     to discover 1-interchange routes.
  3. Package each path as a Route with structured Leg objects.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx

from modules.graph_builder import RailwayGraph


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class Leg:
    train_id:     int
    train_name:   str
    train_type:   str
    from_station: str
    from_name:    str
    to_station:   str
    to_name:      str
    dep_time:     str
    arr_time:     str
    dep_abs:      int
    arr_abs:      int
    travel_min:   int
    days:         str


@dataclass
class Route:
    legs:               List[Leg]
    total_min:          int
    num_interchanges:   int
    interchange_stations: List[str]
    total_distance_km:  float = 0.0

    @property
    def is_direct(self) -> bool:
        return self.num_interchanges == 0

    def summary(self) -> str:
        if self.is_direct:
            return f"Direct  | {self.legs[0].train_name} ({self.legs[0].train_id})"
        parts = " → ".join(
            f"{l.train_name.split()[0]} ({l.train_id})" for l in self.legs
        )
        via = ", ".join(self.interchange_stations)
        return f"Via {via} | {parts}"


# ─── PLANNER ──────────────────────────────────────────────────────────────────

class RoutePlanner:
    """
    Finds up to `max_routes` routes between src and dst.
    Considers:
      - Direct trains (same train both stops)
      - 1-interchange journeys discovered via graph edge traversal
    """

    MIN_CONNECT_MIN = 20   # minimum connection buffer in minutes

    def __init__(self, rg: RailwayGraph):
        self.rg = rg

    # ── public ────────────────────────────────────────────────────────────────

    def find_routes(
        self,
        src: str,
        dst: str,
        max_routes: int = 5,
    ) -> List[Route]:
        routes: List[Route] = []

        # 1. Direct trains
        for direct in self.rg.direct_trains(src, dst):
            routes.append(self._make_direct_route(src, dst, direct))

        # 2. One-interchange routes
        routes.extend(self._one_interchange(src, dst))

        # 3. Two-interchange routes (only if we still have budget)
        if len(routes) < max_routes:
            routes.extend(self._two_interchange(src, dst))

        # Deduplicate by train sequence
        seen: set = set()
        unique: List[Route] = []
        for r in routes:
            key = tuple((l.train_id, l.from_station, l.to_station) for l in r.legs)
            if key not in seen:
                seen.add(key)
                unique.append(r)

        # Sort by total travel time
        unique.sort(key=lambda r: r.total_min)
        return unique[:max_routes]

    # ── internal builders ─────────────────────────────────────────────────────

    def _station_name(self, code: str) -> str:
        info = self.rg.station_info(code)
        return info["station_name"] if info else code

    def _trains_between(self, frm: str, to: str) -> List[dict]:
        """
        Return all trains that travel frm -> to (not necessarily consecutive stops).
        Combines graph edge data with full direct-train lookups.
        """
        results = {
            (d["train_id"], frm, to): d
            for d in self.rg.direct_trains(frm, to)
        }
        # Also grab raw graph edges (consecutive segments)
        edges = self.rg.graph.get_edge_data(frm, to)
        if edges:
            for key, data in edges.items():
                k = (data["train_id"], frm, to)
                if k not in results:
                    results[k] = dict(data)
        return list(results.values())

    def _make_direct_route(self, src: str, dst: str, train_dict: dict) -> Route:
        leg = Leg(
            train_id=train_dict["train_id"],
            train_name=train_dict["train_name"],
            train_type=train_dict["train_type"],
            from_station=src,
            from_name=self._station_name(src),
            to_station=dst,
            to_name=self._station_name(dst),
            dep_time=train_dict["dep_time"],
            arr_time=train_dict["arr_time"],
            dep_abs=train_dict["dep_abs"],
            arr_abs=train_dict["arr_abs"],
            travel_min=train_dict["travel_min"],
            days=train_dict["days"],
        )
        dist = self.rg.heuristic(src, dst) * (80 / 60)  # km approx
        return Route(
            legs=[leg],
            total_min=leg.travel_min,
            num_interchanges=0,
            interchange_stations=[],
            total_distance_km=round(dist, 1),
        )

    def _make_leg(self, d: dict, frm: str, to: str) -> Leg:
        return Leg(
            train_id=d["train_id"],
            train_name=d["train_name"],
            train_type=d.get("train_type", "Express"),
            from_station=frm,
            from_name=self._station_name(frm),
            to_station=to,
            to_name=self._station_name(to),
            dep_time=d["dep_time"],
            arr_time=d["arr_time"],
            dep_abs=d["dep_abs"],
            arr_abs=d["arr_abs"],
            travel_min=d["travel_min"],
            days=d.get("days", "Daily"),
        )

    def _valid_wait(self, arr_abs: int, dep_abs: int) -> Optional[int]:
        """Return wait minutes if the connection is feasible, else None."""
        wait = dep_abs - arr_abs
        if wait < self.MIN_CONNECT_MIN:
            wait += 24 * 60          # Try next-day
        if self.MIN_CONNECT_MIN <= wait <= 8 * 60:
            return wait
        return None

    def _one_interchange(self, src: str, dst: str) -> List[Route]:
        """Find routes with exactly one interchange station.
        Uses _trains_between so non-consecutive leg pairs are captured."""
        routes: List[Route] = []

        # All stations that appear in the schedule
        all_stations = list(self.rg.stations.keys())

        for mid in all_stations:
            if mid in (src, dst):
                continue
            leg1_options = self._trains_between(src, mid)
            leg2_options = self._trains_between(mid, dst)
            if not leg1_options or not leg2_options:
                continue

            for l1d, l2d in itertools.product(leg1_options, leg2_options):
                wait = self._valid_wait(l1d["arr_abs"], l2d["dep_abs"])
                if wait is None:
                    continue
                total_min = l1d["travel_min"] + wait + l2d["travel_min"]
                if total_min > 50 * 60:
                    continue
                routes.append(Route(
                    legs=[self._make_leg(l1d, src, mid), self._make_leg(l2d, mid, dst)],
                    total_min=total_min,
                    num_interchanges=1,
                    interchange_stations=[self._station_name(mid)],
                    total_distance_km=round(
                        self.rg.heuristic(src, mid) * (80/60)
                        + self.rg.heuristic(mid, dst) * (80/60), 1),
                ))
                if len(routes) >= 30:
                    return routes

        return routes

    def _get_edge_legs(self, src: str, dst: str) -> List[dict]:
        """Return all MultiDiGraph edges from src -> dst as dicts."""
        edges = self.rg.graph.get_edge_data(src, dst)
        if not edges:
            return []
        result = []
        for key, data in edges.items():
            result.append(dict(data))
        return result

    def _two_interchange(self, src: str, dst: str) -> List[Route]:
        """Find routes with exactly two interchanges (A -> mid1 -> mid2 -> B)."""
        routes: List[Route] = []

        # mid1 candidates: any station reachable from src
        all_stations = list(self.rg.stations.keys())
        for mid1 in all_stations:
            if mid1 in (src, dst):
                continue
            leg1_opts = self._trains_between(src, mid1)
            if not leg1_opts:
                continue
            for mid2 in all_stations:
                if mid2 in (src, dst, mid1):
                    continue
                leg2_opts = self._trains_between(mid1, mid2)
                leg3_opts = self._trains_between(mid2, dst)
                if not leg2_opts or not leg3_opts:
                    continue

                for l1d, l2d, l3d in itertools.product(leg1_opts, leg2_opts, leg3_opts):
                    w1 = self._valid_wait(l1d["arr_abs"], l2d["dep_abs"])
                    if w1 is None:
                        continue
                    w2 = self._valid_wait(l2d["arr_abs"], l3d["dep_abs"])
                    if w2 is None:
                        continue
                    total_min = l1d["travel_min"] + w1 + l2d["travel_min"] + w2 + l3d["travel_min"]
                    if total_min > 50 * 60:
                        continue
                    routes.append(Route(
                        legs=[
                            self._make_leg(l1d, src,  mid1),
                            self._make_leg(l2d, mid1, mid2),
                            self._make_leg(l3d, mid2, dst),
                        ],
                        total_min=total_min,
                        num_interchanges=2,
                        interchange_stations=[
                            self._station_name(mid1),
                            self._station_name(mid2),
                        ],
                        total_distance_km=round(
                            self.rg.heuristic(src,  mid1) * (80/60)
                            + self.rg.heuristic(mid1, mid2) * (80/60)
                            + self.rg.heuristic(mid2, dst)  * (80/60), 1,
                        ),
                    ))
                    if len(routes) >= 20:
                        return routes

        return routes
