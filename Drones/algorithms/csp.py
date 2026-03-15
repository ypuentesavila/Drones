from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Basic backtracking search without optimizations.

    Tips:
    - An assignment is a dictionary mapping variables to values (e.g. {X1: Cell(1,2), X2: Cell(3,4)}).
    - Use csp.assign(var, value, assignment) to assign a value to a variable.
    - Use csp.unassign(var, assignment) to unassign a variable.
    - Use csp.is_consistent(var, value, assignment) to check if an assignment is consistent with the constraints.
    - Use csp.is_complete(assignment) to check if the assignment is complete (all variables assigned).
    - Use csp.get_unassigned_variables(assignment) to get a list of unassigned variables.
    - Use csp.domains[var] to get the list of possible values for a variable.
    - Use csp.get_neighbors(var) to get the list of variables that share a constraint with var.
    - Add logs to measure how good your implementation is (e.g. number of assignments, backtracks).

    You can find inspiration in the textbook's pseudocode:
    Artificial Intelligence: A Modern Approach (4th Edition) by Russell and Norvig, Chapter 5: Constraint Satisfaction Problems
    """
    def retroceder(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return asignacion.copy()

        no_asignadas = csp.get_unassigned_variables(asignacion)
        if not no_asignadas:
            return asignacion.copy()

        variable = no_asignadas[0]

        for valor in csp.domains[variable][:]:
            if csp.is_consistent(variable, valor, asignacion):
                csp.assign(variable, valor, asignacion)

                resultado = retroceder(asignacion)
                if resultado is not None:
                    return resultado

                csp.unassign(variable, asignacion)

        return None

    return retroceder({})


def backtracking_fc(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with Forward Checking.

    Tips:
    - Forward checking: After assigning a value to a variable, eliminate inconsistent values from
      the domains of unassigned neighbors. If any neighbor's domain becomes empty, backtrack immediately.
    - Save domains before forward checking so you can restore them on backtrack.
    - Use csp.get_neighbors(var) to get variables that share constraints with var.
    - Use csp.is_consistent(neighbor, val, assignment) to check if a value is still consistent.
    - Forward checking reduces the search space by detecting failures earlier than basic backtracking.
    """
    def verificacion_hacia_adelante(
        variable: str, asignacion: dict[str, str]
    ) -> tuple[bool, dict[str, list[str]]]:
        dominios_guardados = {
            vecino: csp.domains[vecino][:]
            for vecino in csp.get_neighbors(variable)
            if vecino not in asignacion
        }

        for vecino in dominios_guardados:
            nuevo_dominio = [
                valor
                for valor in csp.domains[vecino]
                if csp.is_consistent(vecino, valor, asignacion)
            ]
            csp.domains[vecino] = nuevo_dominio

            if len(csp.domains[vecino]) == 0:
                return False, dominios_guardados

        return True, dominios_guardados

    def restaurar_dominios(dominios_guardados: dict[str, list[str]]) -> None:
        for variable, dominio in dominios_guardados.items():
            csp.domains[variable] = dominio

    def retroceder(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return asignacion.copy()

        no_asignadas = csp.get_unassigned_variables(asignacion)
        if not no_asignadas:
            return asignacion.copy()

        variable = no_asignadas[0]

        for valor in csp.domains[variable][:]:
            if csp.is_consistent(variable, valor, asignacion):
                csp.assign(variable, valor, asignacion)

                exito, dominios_guardados = verificacion_hacia_adelante(
                    variable, asignacion
                )
                if exito:
                    resultado = retroceder(asignacion)
                    if resultado is not None:
                        return resultado

                restaurar_dominios(dominios_guardados)
                csp.unassign(variable, asignacion)

        return None

    return retroceder({})


def backtracking_ac3(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking search with AC-3 arc consistency.

    Tips:
    - AC-3 enforces arc consistency: for every pair of constrained variables (Xi, Xj), every value
      in Xi's domain must have at least one supporting value in Xj's domain.
    - Run AC-3 before starting backtracking to reduce domains globally.
    - After each assignment, run AC-3 on arcs involving the assigned variable's neighbors.
    - If AC-3 empties any domain, the current assignment is inconsistent - backtrack.
    - You can create helper functions such as:
      - a values_compatible function to check if two variable-value pairs are consistent with the constraints.
      - a revise function that removes unsupported values from one variable's domain.
      - an ac3 function that manages the queue of arcs to check and calls revise.
      - a backtrack function that integrates AC-3 into the search process.
    """
    def valores_compatibles(
        xi: str, valor_xi: str, xj: str, valor_xj: str, asignacion: dict[str, str]
    ) -> bool:
        asignacion_temporal = asignacion.copy()
        asignacion_temporal[xi] = valor_xi

        if not csp.is_consistent(xi, valor_xi, asignacion):
            return False

        return csp.is_consistent(xj, valor_xj, asignacion_temporal)

    def revisar(xi: str, xj: str, asignacion: dict[str, str]) -> bool:
        revisado = False
        nuevo_dominio = []

        for valor_xi in csp.domains[xi]:
            tiene_soporte = False
            for valor_xj in csp.domains[xj]:
                if valores_compatibles(xi, valor_xi, xj, valor_xj, asignacion):
                    tiene_soporte = True
                    break

            if tiene_soporte:
                nuevo_dominio.append(valor_xi)
            else:
                revisado = True

        if revisado:
            csp.domains[xi] = nuevo_dominio

        return revisado

    def ac3(cola: deque[tuple[str, str]], asignacion: dict[str, str]) -> bool:
        while cola:
            xi, xj = cola.popleft()

            if revisar(xi, xj, asignacion):
                if len(csp.domains[xi]) == 0:
                    return False

                for xk in csp.get_neighbors(xi):
                    if xk != xj:
                        cola.append((xk, xi))

        return True

    def retroceder(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return asignacion.copy()

        no_asignadas = csp.get_unassigned_variables(asignacion)
        if not no_asignadas:
            return asignacion.copy()

        variable = no_asignadas[0]

        for valor in csp.domains[variable][:]:
            if csp.is_consistent(variable, valor, asignacion):
                dominios_guardados = {v: csp.domains[v][:] for v in csp.variables}

                csp.assign(variable, valor, asignacion)
                csp.domains[variable] = [valor]

                cola = deque(
                    (vecino, variable) for vecino in csp.get_neighbors(variable)
                )

                if ac3(cola, asignacion):
                    resultado = retroceder(asignacion)
                    if resultado is not None:
                        return resultado

                for v in csp.variables:
                    csp.domains[v] = dominios_guardados[v]

                csp.unassign(variable, asignacion)

        return None

    dominios_iniciales = {v: csp.domains[v][:] for v in csp.variables}
    cola_inicial = deque()

    for xi in csp.variables:
        for xj in csp.get_neighbors(xi):
            cola_inicial.append((xi, xj))

    if not ac3(cola_inicial, {}):
        for v in csp.variables:
            csp.domains[v] = dominios_iniciales[v]
        return None

    resultado = retroceder({})

    if resultado is None:
        for v in csp.variables:
            csp.domains[v] = dominios_iniciales[v]

    return resultado


def backtracking_mrv_lcv(csp: DroneAssignmentCSP) -> dict[str, str] | None:
    """
    Backtracking with Forward Checking + MRV + LCV.

    Tips:
    - Combine the techniques from backtracking_fc, mrv_heuristic, and lcv_heuristic.
    - MRV (Minimum Remaining Values): Select the unassigned variable with the fewest legal values.
      Tie-break by degree: prefer the variable with the most unassigned neighbors.
    - LCV (Least Constraining Value): When ordering values for a variable, prefer
      values that rule out the fewest choices for neighboring variables.
    - Use csp.get_num_conflicts(var, value, assignment) to count how many values would be ruled out for neighbors if var=value is assigned.
    """
    def seleccionar_variable_mrv(asignacion: dict[str, str]) -> str:
        no_asignadas = csp.get_unassigned_variables(asignacion)

        def clave(variable: str) -> tuple[int, int]:
            tamano_dominio = len(csp.domains[variable])
            grado = sum(
                1
                for vecino in csp.get_neighbors(variable)
                if vecino not in asignacion
            )
            return (tamano_dominio, -grado)

        return min(no_asignadas, key=clave)

    def ordenar_valores_lcv(variable: str, asignacion: dict[str, str]) -> list[str]:
        return sorted(
            csp.domains[variable][:],
            key=lambda valor: csp.get_num_conflicts(variable, valor, asignacion),
        )

    def verificacion_hacia_adelante(
        variable: str, asignacion: dict[str, str]
    ) -> tuple[bool, dict[str, list[str]]]:
        dominios_guardados = {
            vecino: csp.domains[vecino][:]
            for vecino in csp.get_neighbors(variable)
            if vecino not in asignacion
        }

        for vecino in dominios_guardados:
            nuevo_dominio = [
                valor
                for valor in csp.domains[vecino]
                if csp.is_consistent(vecino, valor, asignacion)
            ]
            csp.domains[vecino] = nuevo_dominio

            if len(csp.domains[vecino]) == 0:
                return False, dominios_guardados

        return True, dominios_guardados

    def restaurar_dominios(dominios_guardados: dict[str, list[str]]) -> None:
        for variable, dominio in dominios_guardados.items():
            csp.domains[variable] = dominio

    def retroceder(asignacion: dict[str, str]) -> dict[str, str] | None:
        if csp.is_complete(asignacion):
            return asignacion.copy()

        variable = seleccionar_variable_mrv(asignacion)

        for valor in ordenar_valores_lcv(variable, asignacion):
            if csp.is_consistent(variable, valor, asignacion):
                csp.assign(variable, valor, asignacion)

                exito, dominios_guardados = verificacion_hacia_adelante(
                    variable, asignacion
                )
                if exito:
                    resultado = retroceder(asignacion)
                    if resultado is not None:
                        return resultado

                restaurar_dominios(dominios_guardados)
                csp.unassign(variable, asignacion)

        return None

    return retroceder({})