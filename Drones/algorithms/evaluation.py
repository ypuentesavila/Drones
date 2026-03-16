from __future__ import annotations

import random
from math import inf
from typing import TYPE_CHECKING

from algorithms.utils import bfs_distance, dijkstra

if TYPE_CHECKING:
    from world.game_state import GameState


def evaluation_function(state: GameState) -> float:
    """
    Evaluation function for non-terminal states of the drone vs. hunters game.

    A good evaluation function can consider multiple factors, such as:
      (a) BFS distance from drone to nearest delivery point (closer is better).
          Uses actual path distance so walls and terrain are respected.
      (b) BFS distance from each hunter to the drone, traversing only normal
          terrain ('.' / ' ').  Hunters blocked by mountains, fog, or storms
          are treated as unreachable (distance = inf) and pose no threat.
      (c) BFS distance to a "safe" position (i.e., a position that is not in the path of any hunter).
      (d) Number of pending deliveries (fewer is better).
      (e) Current score (higher is better).
      (f) Delivery urgency: reward the drone for being close to a delivery it can
          reach strictly before any hunter, so it commits to nearby pickups
          rather than oscillating in place out of excessive hunter fear.
      (g) Adding a revisit penalty can help prevent the drone from getting stuck in cycles.

    Returns a value in [-1000, +1000].

    Tips:
    - Use state.get_drone_position() to get the drone's current (x, y) position.
    - Use state.get_hunter_positions() to get the list of hunter (x, y) positions.
    - Use state.get_pending_deliveries() to get the set of pending delivery (x, y) positions.
    - Use state.get_score() to get the current game score.
    - Use state.get_layout() to get the current layout.
    - Use state.is_win() and state.is_lose() to check terminal states.
    - Use bfs_distance(layout, start, goal, hunter_restricted) from algorithms.utils
      for cached BFS distances. hunter_restricted=True for hunter-only terrain.
    - Use dijkstra(layout, start, goal) from algorithms.utils for cached
      terrain-weighted shortest paths, returning (cost, path).
    - Consider edge cases: no pending deliveries, no hunters nearby.
    - A good evaluation function balances delivery progress with hunter avoidance.
    """
    if state.is_win():
        return 1000.0
    if state.is_lose():
        return -1000.0

    drone_pos = state.get_drone_position()
    hunter_positions = state.get_hunter_positions()
    pending_deliveries = list(state.get_pending_deliveries())
    layout = state.get_layout()

    valor = 0.0

    valor += 8.0 * state.get_score()
    valor -= 120.0 * len(pending_deliveries)

    distancia_entrega_min = inf
    mejor_bono_urgencia = 0.0

    for entrega in pending_deliveries:
        costo_dron, _ = dijkstra(layout, drone_pos, entrega)

        if costo_dron < distancia_entrega_min:
            distancia_entrega_min = costo_dron

        distancia_cazador_min = inf
        for cazador in hunter_positions:
            distancia_cazador = bfs_distance(
                layout, cazador, entrega, hunter_restricted=True
            )
            distancia_cazador_min = min(distancia_cazador_min, distancia_cazador)

        if costo_dron < inf:
            if distancia_cazador_min == inf:
                mejor_bono_urgencia = max(
                    mejor_bono_urgencia,
                    80.0 / (1.0 + costo_dron),
                )
            elif costo_dron < distancia_cazador_min:
                mejor_bono_urgencia = max(
                    mejor_bono_urgencia,
                    120.0 / (1.0 + costo_dron),
                )

    if distancia_entrega_min < inf:
        valor -= 10.0 * distancia_entrega_min

    valor += mejor_bono_urgencia

    distancia_cazador_dron_min = inf
    for cazador in hunter_positions:
        distancia = bfs_distance(
            layout, cazador, drone_pos, hunter_restricted=True
        )
        distancia_cazador_dron_min = min(distancia_cazador_dron_min, distancia)

    if distancia_cazador_dron_min < inf:
        if distancia_cazador_dron_min == 0:
            return -1000.0
        if distancia_cazador_dron_min == 1:
            valor -= 500.0
        elif distancia_cazador_dron_min == 2:
            valor -= 220.0
        elif distancia_cazador_dron_min == 3:
            valor -= 100.0
        else:
            valor += 12.0 * distancia_cazador_dron_min

    # (g) Perturbacion aleatoria para romper ciclos
    valor += random.uniform(-5.0, 5.0)

    return max(-1000.0, min(1000.0, valor))