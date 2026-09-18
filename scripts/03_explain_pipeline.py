"""Sesion 1: ilustrar paso a paso como se detecta una matricula.

Material de apoyo para la memoria. Cubre el Obj2 del enunciado (morfologia en
pre- y post-proceso) mostrando el efecto de cada operacion, y deja explicito el
criterio por el que cada contorno se acepta o se descarta.

Genera en `out/explain/<imagen>/`:
  - pasos.jpg      panel con las seis etapas de `plate_mask()`
  - contornos.jpg  los contornos numerados, verde si pasan y rojo si no

y por consola la tabla de decision contorno a contorno.

Reutiliza `plate_mask(steps=...)` y `candidate_verdict()` de `alpr.detection`,
asi que lo que ilustra es siempre el pipeline real, no una copia.

Uso:
    python scripts/03_explain_pipeline.py                       # imagen por defecto
    python scripts/03_explain_pipeline.py real_plates/Frontal/0216KZP.jpg
    python scripts/03_explain_pipeline.py --zoom 3
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from alpr.detection import (STEP_LABELS, SEARCH_TOP, WORK_WIDTH, aspect_ratio,
                            candidate_verdict, normalized_angle, plate_mask)

DEFAULT_IMAGE = config.REAL_PLATES_DIR / "Lateral" / "0182GLK.jpg"
GREEN, RED = (0, 220, 0), (0, 0, 255)


def labelled(image, text, zoom):
    """Imagen ampliada con un rotulo negro arriba."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(image, (0, 0), (image.shape[1], 15 * zoom), (0, 0, 0), -1)
    cv2.putText(image, text, (5 * zoom, 11 * zoom), cv2.FONT_HERSHEY_SIMPLEX,
                .3 * zoom, (255, 255, 255), max(1, zoom), cv2.LINE_AA)
    return image


def steps_panel(steps, zoom):
    """Panel de 2 columnas con las etapas, en el orden de STEP_LABELS."""
    tiles = [labelled(steps[key], text, zoom) for key, text in STEP_LABELS if key in steps]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles) - 1, 2)]
    return np.vstack(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", nargs="?", default=str(DEFAULT_IMAGE),
                        help="imagen a explicar (por defecto, una lateral del dataset)")
    parser.add_argument("--zoom", type=int, default=2,
                        help="factor de ampliacion de las figuras (por defecto 2)")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.is_absolute():
        path = config.ROOT / path
    image = cv2.imread(str(path))
    if image is None:
        print(f"ERROR: no puedo leer {path}")
        return 1

    # Mismo preproceso que detectPlates: se trabaja sobre la imagen reducida.
    scale = WORK_WIDTH / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    steps = {}
    mask = plate_mask(gray, steps=steps)

    out_dir = config.OUT_DIR / "explain" / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / "pasos.jpg"), steps_panel(steps, args.zoom))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = len(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:SEARCH_TOP]

    print(f"\n{path.name}  ({image.shape[1]}x{image.shape[0]} -> "
          f"{small.shape[1]}x{small.shape[0]} para trabajar)")
    print(f"findContours encuentra {total} manchas blancas; se examinan las "
          f"{len(contours)} mayores.\n")
    print(f"{'#':>2} {'px':>6} {'rect':>6} {'relleno':>8} {'area':>7} "
          f"{'aspecto':>8} {'angulo':>8}  veredicto")
    print("-" * 82)

    canvas = cv2.resize(small, None, fx=args.zoom, fy=args.zoom,
                        interpolation=cv2.INTER_NEAREST)
    small_area = small.shape[0] * small.shape[1]
    n_accepted = 0

    for i, contour in enumerate(contours, 1):
        rect = cv2.minAreaRect(contour)
        accepted, reason = candidate_verdict(rect, small_area)
        n_accepted += accepted

        px = cv2.contourArea(contour)
        rect_area = rect[1][0] * rect[1][1]
        fill = px / rect_area if rect_area else 0
        print(f"{i:>2} {px:>6.0f} {rect_area:>6.0f} {fill:>7.0%} "
              f"{rect_area / small_area:>6.2%} {aspect_ratio(rect):>8.2f} "
              f"{normalized_angle(rect):>7.1f}d  "
              f"{'CANDIDATO' if accepted else 'descartado: ' + reason}")

        color = GREEN if accepted else RED
        box = np.round(cv2.boxPoints(rect) * args.zoom).astype(np.int32)
        cv2.drawContours(canvas, [box], -1, color, max(1, args.zoom))
        cv2.putText(canvas, str(i), tuple(box[box[:, 1].argmin()] + [2, -4]),
                    cv2.FONT_HERSHEY_SIMPLEX, .3 * args.zoom, color,
                    max(1, args.zoom), cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "contornos.jpg"), canvas)

    print(f"\n{n_accepted} candidatos de {len(contours)} examinados.")
    print("El elegido es el de MAS PIXELES (cv2.contourArea), no el de mejor aspecto:")
    print("una matricula es un bloque macizo, y los falsos positivos suelen ser")
    print("tiras finas e inclinadas cuyo rectangulo envolvente engana.")
    print(f"\nverde = pasa los filtros, rojo = descartado")
    print(f"figuras -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
