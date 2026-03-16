import subprocess
import sys
import re

# mismos 10 layouts usados en el experimento de victorias
layouts_test = [
    "tiny_hunt",
    "small_hunt",
    "contested_territory",
    "drone_escape",
    "jungle_chase",
    "tunnel_escape",
    "bottleneck",
    "gauntlet",
    "hunter_cage",
    "open_field"
]

agents = ["MinimaxAgent", "AlphaBetaAgent"]
depth = "3"
n_runs = 1

output_file = "resultados_nodos.txt"

with open(output_file, "w", encoding="utf-8") as f:
    header = "COMPARACION DE NODOS EXPLORADOS | depth=3\n" + "="*60 + "\n"
    print(header)
    f.write(header)

    for layout in layouts_test:
        layout_header = f"\nLAYOUT: {layout}\n" + "-"*40 + "\n"
        print(layout_header, end="")
        f.write(layout_header)

        for agent in agents:
            label = f"  {agent} | depth={depth}\n"
            print(label, end="")
            f.write(label)

            try:
                result = subprocess.run(
                    [
                        sys.executable, "main.py",
                        "-m", "adversarial",
                        "-a", agent,
                        "-l", layout,
                        "-d", depth,
                        "-n", str(n_runs),
                        "-q",
                    ],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=120,
                )

                output = result.stdout + result.stderr

                patron = r"Nodos explorados este turno: (\d+)"
                nodos = re.findall(patron, output)

                if nodos:
                    nodos_int = [int(n) for n in nodos]
                    avg = sum(nodos_int) / len(nodos_int)
                    total = sum(nodos_int)
                    max_n = max(nodos_int)

                    resumen = (
                        f"    Turnos jugados: {len(nodos)} | "
                        f"Promedio/turno: {avg:.0f} | "
                        f"Max/turno: {max_n} | "
                        f"Total: {total}\n"
                    )
                else:
                    resumen = "    (sin datos de nodos)\n"

                print(resumen, end="")
                f.write(resumen)

            except subprocess.TimeoutExpired:
                msg = "    TIMEOUT\n"
                print(msg, end="")
                f.write(msg)

print(f"\nDONE - resultados guardados en {output_file}")