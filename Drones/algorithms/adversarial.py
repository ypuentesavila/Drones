from __future__ import annotations

import random
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import algorithms.evaluation as evaluation
from world.game import Agent, Directions

if TYPE_CHECKING:
    from world.game_state import GameState


class MultiAgentSearchAgent(Agent, ABC):
    """
    Base class for multi-agent search agents (Minimax, AlphaBeta, Expectimax).
    """

    def __init__(self, depth: str = "2", _index: int = 0, prob: str = "0.0") -> None:
        self.index = 0  # Drone is always agent 0
        self.depth = int(depth)
        self.prob = float(
            prob
        )  # Probability that each hunter acts randomly (0=greedy, 1=random)
        self.evaluation_function = evaluation.evaluation_function

    @abstractmethod
    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone from the current GameState.
        """
        pass


class RandomAgent(MultiAgentSearchAgent):
    """
    Agent that chooses a legal action uniformly at random.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Get a random legal action for the drone.
        """
        legal_actions = state.get_legal_actions(self.index)
        return random.choice(legal_actions) if legal_actions else None


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for the drone (MAX) vs hunters (MIN) game.
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using minimax.

        Tips:
        - The game tree alternates: drone (MAX) -> hunter1 (MIN) -> hunter2 (MIN) -> ... -> drone (MAX) -> ...
        - Use self.depth to control the search depth. depth=1 means the drone moves once and each hunter moves once.
        - Use state.get_legal_actions(agent_index) to get legal actions for a specific agent.
        - Use state.generate_successor(agent_index, action) to get the successor state after an action.
        - Use state.is_win() and state.is_lose() to check terminal states.
        - Use state.get_num_agents() to get the total number of agents.
        - Use self.evaluation_function(state) to evaluate leaf/terminal states.
        - The next agent is (agent_index + 1) % num_agents. Depth decreases after all agents have moved (full ply).
        - Return the ACTION (not the value) that maximizes the minimax value for the drone.
        """
        num_agents = state.get_num_agents()

        def minimax(
            current_state: GameState, profundidad: int, agent_index: int
        ) -> float:
            if (
                current_state.is_win()
                or current_state.is_lose()
                or profundidad == self.depth
            ):
                return self.evaluation_function(current_state)

            acciones_legales = current_state.get_legal_actions(agent_index)
            if not acciones_legales:
                return self.evaluation_function(current_state)

            siguiente_agente = (agent_index + 1) % num_agents

            if agent_index == 0:
                mejor_valor = float("-inf")

                for accion in acciones_legales:
                    sucesor = current_state.generate_successor(agent_index, accion)
                    siguiente_profundidad = (
                        profundidad + 1 if siguiente_agente == 0 else profundidad
                    )
                    valor = minimax(sucesor, siguiente_profundidad, siguiente_agente)
                    mejor_valor = max(mejor_valor, valor)

                return mejor_valor

            peor_valor = float("inf")

            for accion in acciones_legales:
                sucesor = current_state.generate_successor(agent_index, accion)
                siguiente_profundidad = (
                    profundidad + 1 if siguiente_agente == 0 else profundidad
                )
                valor = minimax(sucesor, siguiente_profundidad, siguiente_agente)
                peor_valor = min(peor_valor, valor)

            return peor_valor

        acciones_legales = state.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")

        for accion in acciones_legales:
            sucesor = state.generate_successor(self.index, accion)
            siguiente_agente = (self.index + 1) % num_agents
            siguiente_profundidad = 1 if siguiente_agente == 0 else 0
            valor = minimax(sucesor, siguiente_profundidad, siguiente_agente)

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Alpha-Beta pruning agent. Same as Minimax but with alpha-beta pruning.
    MAX node: prune when value > beta (strict).
    MIN node: prune when value < alpha (strict).
    """

    def get_action(self, state: GameState) -> Directions | None:
        """
        Returns the best action for the drone using alpha-beta pruning.

        Tips:
        - Same structure as MinimaxAgent, but with alpha-beta pruning.
        - Alpha: best value MAX can guarantee (initially -inf).
        - Beta: best value MIN can guarantee (initially +inf).
        - MAX node: prune when value > beta (strict inequality, do NOT prune on equality).
        - MIN node: prune when value < alpha (strict inequality, do NOT prune on equality).
        - Update alpha at MAX nodes: alpha = max(alpha, value).
        - Update beta at MIN nodes: beta = min(beta, value).
        - Pass alpha and beta through the recursive calls.
        """
        num_agents = state.get_num_agents()

        def alphabeta(
            current_state: GameState,
            profundidad: int,
            agent_index: int,
            alpha: float,
            beta: float,
        ) -> float:
            if (
                current_state.is_win()
                or current_state.is_lose()
                or profundidad == self.depth
            ):
                return self.evaluation_function(current_state)

            acciones_legales = current_state.get_legal_actions(agent_index)
            if not acciones_legales:
                return self.evaluation_function(current_state)

            siguiente_agente = (agent_index + 1) % num_agents

            if agent_index == 0:
                valor = float("-inf")

                for accion in acciones_legales:
                    sucesor = current_state.generate_successor(agent_index, accion)
                    siguiente_profundidad = (
                        profundidad + 1 if siguiente_agente == 0 else profundidad
                    )
                    valor = max(
                        valor,
                        alphabeta(
                            sucesor,
                            siguiente_profundidad,
                            siguiente_agente,
                            alpha,
                            beta,
                        ),
                    )

                    if valor > beta:
                        return valor

                    alpha = max(alpha, valor)

                return valor

            valor = float("inf")

            for accion in acciones_legales:
                sucesor = current_state.generate_successor(agent_index, accion)
                siguiente_profundidad = (
                    profundidad + 1 if siguiente_agente == 0 else profundidad
                )
                valor = min(
                    valor,
                    alphabeta(
                        sucesor,
                        siguiente_profundidad,
                        siguiente_agente,
                        alpha,
                        beta,
                    ),
                )

                if valor < alpha:
                    return valor

                beta = min(beta, valor)

            return valor

        acciones_legales = state.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")
        alpha = float("-inf")
        beta = float("inf")

        for accion in acciones_legales:
            sucesor = state.generate_successor(self.index, accion)
            siguiente_agente = (self.index + 1) % num_agents
            siguiente_profundidad = 1 if siguiente_agente == 0 else 0
            valor = alphabeta(
                sucesor,
                siguiente_profundidad,
                siguiente_agente,
                alpha,
                beta,
            )

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

            alpha = max(alpha, mejor_valor)

        return mejor_accion


class ExpectimaxAgent(MultiAgentSearchAgent):

    def get_action(self, estado: GameState) -> Directions | None:

        def valor_expectimax(
            estado_actual: GameState,
            profundidad: int,
            indice_agente: int
        ) -> float:

            if profundidad == 0 or estado_actual.is_win() or estado_actual.is_lose():
                return self.evaluation_function(estado_actual)

            acciones_legales = estado_actual.get_legal_actions(indice_agente)
            if not acciones_legales:
                return self.evaluation_function(estado_actual)

            numero_agentes = estado_actual.get_num_agents()
            siguiente_agente = (indice_agente + 1) % numero_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            # Nodo MAX (drone)
            if indice_agente == 0:
                valor = float("-inf")

                for accion in acciones_legales:
                    sucesor = estado_actual.generate_successor(indice_agente, accion)

                    valor = max(
                        valor,
                        valor_expectimax(
                            sucesor,
                            siguiente_profundidad,
                            siguiente_agente
                        )
                    )

                return valor

            # Nodo CHANCE (hunters con modelo mixto)
            valores_hijos = []

            for accion in acciones_legales:
                sucesor = estado_actual.generate_successor(indice_agente, accion)

                valores_hijos.append(
                    valor_expectimax(
                        sucesor,
                        siguiente_profundidad,
                        siguiente_agente
                    )
                )

            parte_codiciosa = min(valores_hijos)
            parte_aleatoria = sum(valores_hijos) / len(valores_hijos)

            return (1 - self.prob) * parte_codiciosa + self.prob * parte_aleatoria

        acciones_legales = estado.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")

        numero_agentes = estado.get_num_agents()
        siguiente_agente = (self.index + 1) % numero_agentes
        siguiente_profundidad = self.depth - 1 if siguiente_agente == 0 else self.depth

        for accion in acciones_legales:

            sucesor = estado.generate_successor(self.index, accion)

            valor = valor_expectimax(
                sucesor,
                siguiente_profundidad,
                siguiente_agente
            )

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion
















