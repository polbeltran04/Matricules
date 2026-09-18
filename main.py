"""Punto de entrada unico: ejecuta el pipeline completo del challenge.

Cada etapa sigue siendo un script independiente y ejecutable por su cuenta
(el enunciado evalua los entregables por separado); esto solo los encadena en
orden, corta si alguno falla y resume al final que se ha generado.

Uso:
    python main.py                    # todo: tests + las tres etapas
    python main.py explore detect     # solo las etapas indicadas
    python main.py --new              # incluye new_plates/ en la exploracion
    python main.py -n 5               # limita las imagenes anotadas de `detect`
    python main.py --list             # que etapas hay
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import config

ROOT = Path(__file__).resolve().parent

# (nombre, script, descripcion). El orden es el de ejecucion.
STAGES = [
    ("tests", "tests/test_detection.py", "Tests de alpr.detection"),
    ("protocol", "scripts/00_acquisition_protocol.py", "Protocolo de adquisicion (EXIF)"),
    ("explore", "scripts/01_data_exploration.py", "Propiedades, estadistica y figuras"),
    ("detect", "scripts/02_show_detections.py", "Bounding boxes y mosaicos"),
    ("explain", "scripts/03_explain_pipeline.py", "Pipeline paso a paso (figuras memoria)"),
]

STAGE_NAMES = [name for name, _, _ in STAGES]


def stage_args(name, args):
    """Flags propios de cada etapa. Solo algunas aceptan opciones."""
    if name == "explore" and args.new:
        return ["--new"]
    if name == "detect" and args.limit:
        return ["-n", str(args.limit)]
    return []


def run_stage(name, script, description, extra):
    print(f"\n{'=' * 70}\n  {name.upper()}  -  {description}\n{'=' * 70}", flush=True)
    start = time.perf_counter()
    # Sin capturar la salida: interesa ver el progreso de cada script en vivo.
    code = subprocess.run([sys.executable, str(ROOT / script), *extra], cwd=ROOT).returncode
    return code, time.perf_counter() - start


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stages", nargs="*", metavar="ETAPA",
                        help=f"etapas a ejecutar ({', '.join(STAGE_NAMES)}); "
                             "por defecto, todas")
    parser.add_argument("--new", action="store_true",
                        help="incluir new_plates/ en la exploracion")
    parser.add_argument("-n", "--limit", type=int, metavar="N",
                        help="limitar a N imagenes por vista al anotar")
    parser.add_argument("--list", action="store_true", help="listar las etapas y salir")
    args = parser.parse_args()

    if args.list:
        print("Etapas disponibles, en orden de ejecucion:\n")
        for name, script, description in STAGES:
            print(f"  {name:<9} {description:<38} {script}")
        return 0

    unknown = [s for s in args.stages if s not in STAGE_NAMES]
    if unknown:
        parser.error(f"etapa desconocida: {', '.join(unknown)}. "
                     f"Disponibles: {', '.join(STAGE_NAMES)}")

    selected = [s for s in STAGES if not args.stages or s[0] in args.stages]

    if not config.REAL_PLATES_DIR.is_dir():
        print(f"ERROR: no encuentro el dataset en {config.REAL_PLATES_DIR}")
        print("Descomprime real_plates/ en la raiz del proyecto (ver README.md).")
        return 1

    config.OUT_DIR.mkdir(exist_ok=True)

    results = []
    for name, script, description in selected:
        code, elapsed = run_stage(name, script, description, stage_args(name, args))
        results.append((name, code, elapsed))
        if code != 0:
            print(f"\n{name} ha fallado (codigo {code}); se detiene el pipeline.")
            break

    print(f"\n{'=' * 70}\n  RESUMEN\n{'=' * 70}")
    for name, code, elapsed in results:
        print(f"  {'OK  ' if code == 0 else 'FALLO'}  {name:<9} {elapsed:6.1f}s")

    failed = [name for name, code, _ in results if code != 0]
    skipped = len(selected) - len(results)
    if skipped:
        print(f"  {skipped} etapa(s) sin ejecutar")
    if not failed:
        print(f"\nTodo correcto. Resultados en {config.OUT_DIR}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
