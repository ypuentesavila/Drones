from __future__ import annotations

from typing import TYPE_CHECKING

from algorithms.utils import bfs_distance, dijkstra


if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.
    """

    if state.is_win():
        return 1000.0

    if state.is_lose():
        return -1000.0

    layout = state.get_layout()
    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = list(state.get_pending_deliveries())
    current_score = float(state.get_score())

    value = 0.0

    value += 10.0 * current_score
    value -= 120.0 * len(pending_deliveries)

    if pending_deliveries:
        delivery_costs = []
        urgency_bonus = 0.0

        for delivery in pending_deliveries:
            cost, _ = dijkstra(layout, drone_pos, delivery)

            if cost == float("inf"):
                value -= 80.0
                continue

            delivery_costs.append(cost)

            value += 35.0 / (1.0 + cost)

            if hunter_positions:
                hunter_dists = [
                    bfs_distance(layout, hunter, delivery, hunter_restricted=True)
                    for hunter in hunter_positions
                ]

                reachable_hunter_dists = [
                    d for d in hunter_dists if d != float("inf")
                ]

                if not reachable_hunter_dists:
                    urgency_bonus += 30.0
                else:
                    nearest_hunter = min(reachable_hunter_dists)
                    if cost < nearest_hunter:
                        urgency_bonus += 20.0 + 5.0 * (nearest_hunter - cost)

        if delivery_costs:
            nearest_delivery = min(delivery_costs)
            avg_delivery = sum(delivery_costs) / len(delivery_costs)

            value -= 15.0 * nearest_delivery
            value -= 3.0 * avg_delivery
            value += urgency_bonus

    if hunter_positions:
        hunter_to_drone = [
            bfs_distance(layout, hunter, drone_pos, hunter_restricted=True)
            for hunter in hunter_positions
        ]

        reachable_hunters = [d for d in hunter_to_drone if d != float("inf")]

        if reachable_hunters:
            nearest_hunter = min(reachable_hunters)

            if nearest_hunter == 0:
                return -1000.0
            if nearest_hunter == 1:
                value -= 500.0
            elif nearest_hunter == 2:
                value -= 250.0
            elif nearest_hunter == 3:
                value -= 120.0
            else:
                value += 8.0 * nearest_hunter

            for dist in reachable_hunters:
                value -= 30.0 / (1.0 + dist)
        else:
            value += 40.0

    if value > 1000.0:
        return 1000.0
    if value < -1000.0:
        return -1000.0

    return value