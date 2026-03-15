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
    def get_action(self, estado: GameState) -> Directions | None:
        def valor_minimax(estado_actual: GameState, profundidad: int, indice_agente: int) -> float:

            if profundidad == 0 or estado_actual.is_win() or estado_actual.is_lose():
                return self.evaluation_function(estado_actual)

            acciones_legales = estado_actual.get_legal_actions(indice_agente)
            if not acciones_legales:
                return self.evaluation_function(estado_actual)

            numero_agentes = estado_actual.get_num_agents()
            siguiente_agente = (indice_agente + 1) % numero_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            # Nodo MAX 
            if indice_agente == 0:
                valor = float("-inf")
                for accion in acciones_legales:
                    sucesor = estado_actual.generate_successor(indice_agente, accion)
                    valor = max(
                        valor,
                        valor_minimax(sucesor, siguiente_profundidad, siguiente_agente)
                    )
                return valor

            # Nodo MIN
            valor = float("inf")
            for accion in acciones_legales:
                sucesor = estado_actual.generate_successor(indice_agente, accion)
                valor = min(
                    valor,
                    valor_minimax(sucesor, siguiente_profundidad, siguiente_agente)
                )
            return valor


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
            valor = valor_minimax(sucesor, siguiente_profundidad, siguiente_agente)

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

        return mejor_accion


class AlphaBetaAgent(MultiAgentSearchAgent):

    def get_action(self, estado: GameState) -> Directions | None:

        def valor_alfabeta(
            estado_actual: GameState,
            profundidad: int,
            indice_agente: int,
            alfa: float,
            beta: float,
        ) -> float:

            if profundidad == 0 or estado_actual.is_win() or estado_actual.is_lose():
                return self.evaluation_function(estado_actual)

            acciones_legales = estado_actual.get_legal_actions(indice_agente)
            if not acciones_legales:
                return self.evaluation_function(estado_actual)

            numero_agentes = estado_actual.get_num_agents()
            siguiente_agente = (indice_agente + 1) % numero_agentes
            siguiente_profundidad = profundidad - 1 if siguiente_agente == 0 else profundidad

            # Nodo MAX
            if indice_agente == 0:
                valor = float("-inf")

                for accion in acciones_legales:
                    sucesor = estado_actual.generate_successor(indice_agente, accion)

                    valor = max(
                        valor,
                        valor_alfabeta(
                            sucesor,
                            siguiente_profundidad,
                            siguiente_agente,
                            alfa,
                            beta,
                        ),
                    )

                    if valor > beta:
                        return valor

                    alfa = max(alfa, valor)

                return valor

            # Nodo MIN
            valor = float("inf")

            for accion in acciones_legales:
                sucesor = estado_actual.generate_successor(indice_agente, accion)

                valor = min(
                    valor,
                    valor_alfabeta(
                        sucesor,
                        siguiente_profundidad,
                        siguiente_agente,
                        alfa,
                        beta,
                    ),
                )

                if valor < alfa:
                    return valor

                beta = min(beta, valor)

            return valor


        acciones_legales = estado.get_legal_actions(self.index)
        if not acciones_legales:
            return None

        mejor_accion = None
        mejor_valor = float("-inf")

        alfa = float("-inf")
        beta = float("inf")

        numero_agentes = estado.get_num_agents()
        siguiente_agente = (self.index + 1) % numero_agentes
        siguiente_profundidad = self.depth - 1 if siguiente_agente == 0 else self.depth

        for accion in acciones_legales:

            sucesor = estado.generate_successor(self.index, accion)

            valor = valor_alfabeta(
                sucesor,
                siguiente_profundidad,
                siguiente_agente,
                alfa,
                beta,
            )

            if valor > mejor_valor:
                mejor_valor = valor
                mejor_accion = accion

            alfa = max(alfa, mejor_valor)

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
