"""Detour core: batch detour for all points on one walking leg via two Dijkstra runs.

    detour(P) = t(A→P) + t(P→B) − t(A→B)

With dist_A = Dijkstra from A and dist_to_B = Dijkstra from B on the reverse
graph, every point is scored in O(1) after the two runs — no per-point routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import networkx as nx

from .graph import MAX_WALK_MIN, dijkstra_with_prev, reconstruct_path
from .points import PickupPoint


@dataclass
class LegDetourResult:
    """One point scored against one walking leg A→B."""

    point_id: str
    leg_from: str
    leg_to: str
    t_a_to_p_min: float
    t_p_to_b_min: float
    t_a_to_b_min: float
    detour_min: float
    on_shortest_path: bool
    node: int


@dataclass
class LegScoreBoard:
    """Two Dijkstra runs for one leg, reusable for every point."""

    leg_from: str
    leg_to: str
    node_a: int
    node_b: int
    t_a_to_b_min: float
    dist_from_a: dict[int, float]
    dist_to_b: dict[int, float]
    path_nodes: frozenset[int]
    path_coords: list[tuple[float, float]]
    dijkstra_from_a_s: float
    dijkstra_to_b_s: float


def score_leg(
    graph: nx.MultiDiGraph,
    leg_from_id: str,
    leg_to_id: str,
    landmark_nodes: dict[str, int],
) -> LegScoreBoard:
    """Run the two Dijkstra passes for leg A→B and keep the path node set."""
    import time as time_mod

    node_a = landmark_nodes[leg_from_id]
    node_b = landmark_nodes[leg_to_id]

    t0 = time_mod.perf_counter()
    dist_from_a, prev = dijkstra_with_prev(graph, node_a, max_cost=MAX_WALK_MIN)
    t1 = time_mod.perf_counter()
    # reverse view: Dijkstra from B gives dist to B for every node
    dist_to_b, _ = dijkstra_with_prev(
        graph.reverse(copy=False), node_b, max_cost=MAX_WALK_MIN
    )
    t2 = time_mod.perf_counter()

    if node_b not in dist_from_a:
        raise nx.NetworkXNoPath(f"No path {leg_from_id} → {leg_to_id}")
    t_a_to_b = dist_from_a[node_b]
    path = reconstruct_path(prev, node_a, node_b)

    # geometry of the shortest path (for optional plotting)
    from .graph import path_coordinates

    return LegScoreBoard(
        leg_from=leg_from_id,
        leg_to=leg_to_id,
        node_a=node_a,
        node_b=node_b,
        t_a_to_b_min=float(t_a_to_b),
        dist_from_a=dist_from_a,
        dist_to_b=dist_to_b,
        path_nodes=frozenset(path),
        path_coords=path_coordinates(graph, path),
        dijkstra_from_a_s=t1 - t0,
        dijkstra_to_b_s=t2 - t1,
    )


def detour_for_point(board: LegScoreBoard, point: PickupPoint) -> LegDetourResult:
    """Score one point against a precomputed leg board."""
    if point.node is None:
        raise ValueError(f"Point {point.id} is not snapped to the graph")
    n = point.node
    t_ap = board.dist_from_a.get(n, math.inf)
    t_pb = board.dist_to_b.get(n, math.inf)
    if math.isinf(t_ap) or math.isinf(t_pb):
        detour = math.inf
    else:
        detour = t_ap + t_pb - board.t_a_to_b_min
        # numerical safety: triangle inequality ⇒ detour ≥ 0
        if detour < -1e-6:
            raise RuntimeError(
                f"Negative detour {detour} for {point.id} on "
                f"{board.leg_from}→{board.leg_to} (bug in distances)"
            )
        detour = max(0.0, detour)
    return LegDetourResult(
        point_id=point.id,
        leg_from=board.leg_from,
        leg_to=board.leg_to,
        t_a_to_p_min=t_ap,
        t_p_to_b_min=t_pb,
        t_a_to_b_min=board.t_a_to_b_min,
        detour_min=detour,
        on_shortest_path=n in board.path_nodes,
        node=n,
    )


def best_leg_for_point(
    boards: list[LegScoreBoard], point: PickupPoint
) -> LegDetourResult:
    """Attach the point to the leg with the smallest detour (min over legs)."""
    results = [detour_for_point(b, point) for b in boards]
    return min(results, key=lambda r: (math.isinf(r.detour_min), r.detour_min))


def commute_timeline(route) -> list[dict]:
    """Ordered walk/transfer segments from origin to destination.

    Walk legs are chained through transfers by landmark id. Returns events:
      {"kind": "walk"|"ride", "from", "to", "label", "ride_min": float}
    """
    events: list[dict] = []
    remaining_legs = list(route.walk_legs)
    remaining_xfers = list(route.transfers)
    cursor = route.origin

    while remaining_legs or remaining_xfers:
        # next walk leg starting at cursor
        walk = next((L for L in remaining_legs if L.from_id == cursor), None)
        # next transfer starting at cursor (only if no walk starts here,
        # or after a walk has been emitted and we are at a transfer point)
        if walk is not None:
            events.append({
                "kind": "walk",
                "from": walk.from_id,
                "to": walk.to_id,
                "label": walk.label,
                "ride_min": 0.0,
            })
            remaining_legs.remove(walk)
            cursor = walk.to_id
            continue

        xfer = next((T for T in remaining_xfers if T.from_id == cursor), None)
        if xfer is not None:
            events.append({
                "kind": "ride",
                "from": xfer.from_id,
                "to": xfer.to_id,
                "label": xfer.label,
                "ride_min": xfer.ride_min,
            })
            remaining_xfers.remove(xfer)
            cursor = xfer.to_id
            continue

        # not chainable: keep original order as fallback
        if remaining_legs:
            walk = remaining_legs.pop(0)
            events.append({
                "kind": "walk",
                "from": walk.from_id,
                "to": walk.to_id,
                "label": walk.label,
                "ride_min": 0.0,
            })
            cursor = walk.to_id
        else:
            xfer = remaining_xfers.pop(0)
            events.append({
                "kind": "ride",
                "from": xfer.from_id,
                "to": xfer.to_id,
                "label": xfer.label,
                "ride_min": xfer.ride_min,
            })
            cursor = xfer.to_id

    return events


def arrival_offset_min(events: list[dict], leg_from: str, leg_to: str, t_a_to_p_min: float) -> float:
    """Minutes from commute departure until the walker reaches P on leg A→B.

    = walking time of earlier walk legs
      + ride time of transfers before this leg
      + t(A→P)
    """
    offset = 0.0
    for ev in events:
        if ev["kind"] == "walk" and ev["from"] == leg_from and ev["to"] == leg_to:
            return offset + t_a_to_p_min
        if ev["kind"] == "walk":
            # earlier walk leg: use its full duration (filled by caller via dict)
            offset += ev.get("leg_duration_min", 0.0)
        else:
            offset += ev.get("ride_min", 0.0)
    raise ValueError(f"Leg {leg_from}→{leg_to} not found in timeline")
