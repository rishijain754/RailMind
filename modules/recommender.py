"""
recommender.py — Ranks routes and labels them (Best / Fastest / Safest).

Scoring formula:
    score = 0.40 * time_score
          + 0.40 * reliability_score
          + 0.20 * simplicity_score

    time_score        = 1 / total_travel_min          (lower time → better)
    reliability_score = 1 / (1 + avg_predicted_delay) (lower delay → better)
    simplicity_score  = 1 / (1 + num_interchanges)    (fewer hops → better)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from modules.route_planner import Route
from modules.connection_checker import ConnectionStatus
from modules.delay_predictor import DelayPredictor


# ─── DATA CLASSES ─────────────────────────────────────────────────────────────

@dataclass
class RankedRoute:
    rank:               int
    route:              Route
    connections:        List[ConnectionStatus]
    total_delay_min:    float
    avg_confidence:     float
    score:              float
    labels:             List[str] = field(default_factory=list)

    @property
    def overall_status(self) -> str:
        """Worst connection status across all legs."""
        if not self.connections:
            return "N/A"
        priority = {"NOT POSSIBLE": 0, "RISKY": 1, "SAFE": 2}
        worst = min(self.connections, key=lambda c: priority.get(c.status, 2))
        return worst.status

    @property
    def overall_emoji(self) -> str:
        status_map = {"SAFE": "[OK]", "RISKY": "[!!]", "NOT POSSIBLE": "[NO]"}
        return status_map.get(self.overall_status, "[?]")


# ─── RECOMMENDER ──────────────────────────────────────────────────────────────

class Recommender:

    WEIGHT_TIME        = 0.40
    WEIGHT_RELIABILITY = 0.40
    WEIGHT_SIMPLICITY  = 0.20

    def rank(
        self,
        routes: List[Route],
        connections_per_route: List[List[ConnectionStatus]],
    ) -> List[RankedRoute]:
        """
        Rank routes and return RankedRoute objects with labels.

        Parameters
        ----------
        routes                : list of Route objects
        connections_per_route : parallel list of ConnectionStatus lists
        """
        if not routes:
            return []

        candidates: List[RankedRoute] = []
        for route, conns in zip(routes, connections_per_route):
            total_delay = sum(c.predicted_delay for c in conns) if conns else 0.0
            avg_conf    = (
                sum(c.confidence for c in conns) / len(conns)
                if conns else 1.0
            )

            time_score        = 1.0 / max(route.total_min, 1)
            reliability_score = 1.0 / (1.0 + total_delay)
            simplicity_score  = 1.0 / (1.0 + route.num_interchanges)

            score = (
                self.WEIGHT_TIME        * time_score
                + self.WEIGHT_RELIABILITY * reliability_score
                + self.WEIGHT_SIMPLICITY  * simplicity_score
            )

            candidates.append(RankedRoute(
                rank=0,
                route=route,
                connections=conns,
                total_delay_min=round(total_delay, 1),
                avg_confidence=round(avg_conf, 2),
                score=round(score * 1e6, 4),
                labels=[],
            ))

        # Sort by composite score descending
        candidates.sort(key=lambda r: -r.score)
        for i, c in enumerate(candidates):
            c.rank = i + 1

        # Assign special labels
        self._assign_labels(candidates)
        return candidates

    def _assign_labels(self, ranked: List[RankedRoute]):
        if not ranked:
            return

        # Best route = highest overall score
        ranked[0].labels.append("[BEST] Best Route")

        # Fastest = lowest total travel time
        fastest = min(ranked, key=lambda r: r.route.total_min)
        if "[BEST] Best Route" not in fastest.labels:
            fastest.labels.append("[FAST] Fastest")

        # Safest = lowest predicted delay
        safest = min(ranked, key=lambda r: r.total_delay_min)
        if "[BEST] Best Route" not in safest.labels and "[FAST] Fastest" not in safest.labels:
            safest.labels.append("[SAFE] Safest")
        elif "[BEST] Best Route" in safest.labels or "[FAST] Fastest" in safest.labels:
            if "[SAFE] Safest" not in safest.labels:
                safest.labels.append("[SAFE] Safest")

        # Mark remaining as alternatives
        for r in ranked:
            if not r.labels:
                r.labels.append(f"[ALT] Alternative {r.rank - 1}")
