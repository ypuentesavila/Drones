from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from algorithms.problems_csp import DroneAssignmentCSP


def backtracking_search(csp: DroneAssignmentCSP) -> dict[str, str] | None:
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

                cola = deque((vecino, variable) for vecino in csp.get_neighbors(variable))

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