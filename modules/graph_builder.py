"""
graph_builder.py  — Builds a directed multi-graph of the Indian railway network.

Nodes  = station codes
Edges  = direct train segments between consecutive stops
         (one edge per train per segment)
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Tuple, Optional

import pandas as pd
import networkx as nx


# ─── HAVERSINE ────────────────────────────────────────────────────────────────

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    φ1, φ2 = math.radians(lat1), math.radians(lat2)
    dφ = math.radians(lat2 - lat1)
    dλ = math.radians(lon2 - lon1)
    a = math.sin(dφ / 2) ** 2 + math.cos(φ1) * math.cos(φ2) * math.sin(dλ / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ─── RAILWAY GRAPH ────────────────────────────────────────────────────────────

class RailwayGraph:
    """
    Builds and exposes a NetworkX MultiDiGraph from train schedule CSVs.

    graph   — MultiDiGraph  (nodes=stations, edges=train segments)
    station — dict[code -> row dict]
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.stations: Dict[str, dict] = {}
        self._schedule: Optional[pd.DataFrame] = None

    # ── build ─────────────────────────────────────────────────────────────────

    def build(self) -> "RailwayGraph":
        stn_path = os.path.join(self.data_dir, "stations.csv")
        sch_path = os.path.join(self.data_dir, "train_schedule.csv")

        if not os.path.exists(stn_path) or not os.path.exists(sch_path):
            raise FileNotFoundError(
                "Data files not found. Run `python main.py generate-data` first."
            )

        stn_df = pd.read_csv(stn_path)
        sch_df = pd.read_csv(sch_path)
        self._schedule = sch_df

        # ── add station nodes ─────────────────────────────────────────────────
        for _, row in stn_df.iterrows():
            code = row["station_code"]
            info = row.to_dict()
            self.stations[code] = info
            self.graph.add_node(code, **info)

        # ── add edge for every consecutive pair of stops within a train ───────
        for tid, grp in sch_df.groupby("train_id"):
            grp = grp.sort_values("stop_number").reset_index(drop=True)
            for i in range(len(grp) - 1):
                src_row = grp.iloc[i]
                dst_row = grp.iloc[i + 1]

                src = src_row["station_code"]
                dst = dst_row["station_code"]

                travel_min = int(dst_row["arr_abs_min"]) - int(src_row["dep_abs_min"])
                if travel_min <= 0:
                    travel_min = 1  # safety guard

                self.graph.add_edge(
                    src, dst,
                    key=str(tid),
                    train_id=int(tid),
                    train_name=src_row["train_name"],
                    train_type=src_row["train_type"],
                    dep_time=src_row["departure_time"],
                    arr_time=dst_row["arrival_time"],
                    dep_abs=int(src_row["dep_abs_min"]),
                    arr_abs=int(dst_row["arr_abs_min"]),
                    travel_min=travel_min,
                    days=src_row["days"],
                    weight=travel_min,
                )

        return self

    # ── query helpers ─────────────────────────────────────────────────────────

    def station_info(self, code: str) -> Optional[dict]:
        return self.stations.get(code)

    def get_lat_lon(self, code: str) -> Tuple[float, float]:
        info = self.station_info(code)
        if info:
            return float(info["latitude"]), float(info["longitude"])
        return (0.0, 0.0)

    def heuristic(self, u: str, v: str) -> float:
        """A* heuristic: straight-line km distance (admissible)."""
        lat1, lon1 = self.get_lat_lon(u)
        lat2, lon2 = self.get_lat_lon(v)
        km = haversine_km(lat1, lon1, lat2, lon2)
        # Convert km to approximate minutes at average 80 km/h
        return (km / 80.0) * 60.0

    def direct_trains(self, src: str, dst: str) -> List[dict]:
        """Return all direct train edges between src and dst (non-consecutive ok)."""
        results = []
        if src not in self.graph or dst not in self.graph:
            return results
        # Walk full schedule for each train that visits both nodes in order
        if self._schedule is None:
            return results
        for tid, grp in self._schedule.groupby("train_id"):
            grp = grp.sort_values("stop_number")
            codes = list(grp["station_code"])
            if src in codes and dst in codes:
                si, di = codes.index(src), codes.index(dst)
                if si < di:
                    src_row = grp.iloc[si]
                    dst_row = grp.iloc[di]
                    travel = int(dst_row["arr_abs_min"]) - int(src_row["dep_abs_min"])
                    results.append({
                        "train_id":   int(tid),
                        "train_name": src_row["train_name"],
                        "train_type": src_row["train_type"],
                        "dep_time":   src_row["departure_time"],
                        "arr_time":   dst_row["arrival_time"],
                        "dep_abs":    int(src_row["dep_abs_min"]),
                        "arr_abs":    int(dst_row["arr_abs_min"]),
                        "travel_min": travel,
                        "days":       src_row["days"],
                    })
        return results

    def station_list(self) -> List[dict]:
        return list(self.stations.values())

    def train_schedule(self, train_id: int) -> Optional[pd.DataFrame]:
        if self._schedule is None:
            return None
        grp = self._schedule[self._schedule["train_id"] == train_id]
        return grp.sort_values("stop_number") if not grp.empty else None

    def all_train_ids(self) -> List[int]:
        if self._schedule is None:
            return []
        return sorted(self._schedule["train_id"].unique().tolist())
