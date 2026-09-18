"""Tasa de acierto real del detector, no solo cobertura.

`01_data_exploration.py` mide COBERTURA: en cuantas imagenes `detectPlates`
devolvio algo. Eso no dice si ese algo era la matricula. Este script mide el
ACIERTO, cruzando la deteccion con veredictos revisados a mano.

Como se obtuvieron los veredictos (`validation/detection_verdicts.csv`):

  1. `02_show_detections.py` guarda el recorte del candidato #1 de cada imagen
     en `out/detections/crops/`, nombrado con la matricula del ground truth.
  2. Se revisa cada recorte comparando lo que pone la placa con ese nombre.
     Acierto = el recorte contiene la matricula completa y legible.

No es IoU: no dice si la caja encaja bien, solo si es la matricula y esta
entera. Para IoU harian falta cajas anotadas, que son las mismas que YOLO
necesitara para entrenar en la Sesion 2.

El intervalo de confianza es el de Wilson al 95%, que aguanta bien las
proporciones cercanas a 0 o a 1 y las muestras pequenas.

Uso:
    python scripts/05_detection_accuracy.py
"""

import csv
import sys
from math import sqrt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

VERDICTS = config.ROOT / "validation" / "detection_verdicts.csv"


def wilson(hits, total, z=1.96):
    """Intervalo de confianza de Wilson al 95% para una proporcion."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = hits / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    margin = z * sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return p, max(0.0, centre - margin), min(1.0, centre + margin)


def report(label, rows, width=26):
    hits = sum(1 for r in rows if r["hit"] == "yes")
    p, lo, hi = wilson(hits, len(rows))
    print(f"  {label:<{width}} {hits:>4}/{len(rows):<4} {p:>6.1%}   "
          f"[{lo:.1%} - {hi:.1%}]")


def main():
    if not VERDICTS.exists():
        print(f"ERROR: no encuentro {VERDICTS}")
        print("Se genera revisando los recortes de out/detections/crops/.")
        return 1

    with open(VERDICTS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print(f"ACIERTO DEL DETECTOR  ({len(rows)} imagenes revisadas a mano)\n")
    print(f"  {'grupo':<26} {'aciertos':>9} {'tasa':>8}   IC 95%")
    print("  " + "-" * 62)
    report("TOTAL", rows)
    print()
    for source in sorted({r["source"] for r in rows}):
        subset = [r for r in rows if r["source"] == source]
        report(source, subset)
        for view in sorted({r["view"] for r in subset}):
            report(f"  {view}", [r for r in subset if r["view"] == view])

    issues = [r for r in rows if r["gt_issue"]]
    if issues:
        print(f"\n  Discrepancias de ground truth ({len(issues)}):")
        for r in issues:
            print(f"    {r['source']}/{r['view']}/{r['file']}: {r['gt_issue']}")

    print("\n  Nota: acierto = el recorte del candidato #1 contiene la matricula")
    print("  entera y legible. No es IoU; no mide como de ajustada esta la caja.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
