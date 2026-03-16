import subprocess
import sys

# Solo 10 layouts representativos
layouts = [
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

depth = "2"     # recomendado para correr muchas partidas
n_runs = 10     # 10 corridas para calcular tasas de victoria

output_file = "resultados_adversarial.txt"

with open(output_file, "w", encoding="utf-8") as f:
    for agent in agents:
        header = f"\n{'='*60}\nAGENTE: {agent} | depth={depth} | runs={n_runs}\n{'='*60}\n"
        print(header, end="")
        f.write(header)

        for layout in layouts:
            label = f"\n===== {layout} =====\n"
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
                    timeout=90,
                )

                output = result.stdout + result.stderr

            except subprocess.TimeoutExpired:
                output = "TIMEOUT - layout demasiado complejo para este agente/profundidad\n"

            print(output, end="")
            f.write(output)

print(f"\nDONE - resultados guardados en {output_file}")