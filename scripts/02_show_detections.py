"""Sesion 1: dibujar las bounding boxes candidatas sobre las imagenes.

Tercer bullet de la slide 16 del enunciado ("Show the candidate bounding boxes
on the image"), que el codigo base no llegaba a hacer.

Guarda en `out/detections/<vista>/` una copia de cada imagen con el minAreaRect
de cada candidato y su angulo anotado, y un mosaico con el recorte del mejor
candidato de cada imagen para revisar de un vistazo si la deteccion acierta.

Uso:
    python scripts/02_show_detections.py             # todas las imagenes
    python scripts/02_show_detections.py -n 8        # solo las 8 primeras/vista
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from alpr.dataset import draw_candidates, unique_images
from alpr.detection import detectPlates, normalized_angle

PREVIEW_WIDTH = 1024        # ancho al que se guardan las imagenes anotadas
CROP_SIZE = (240, 60)       # tamano de cada recorte del mosaico


def crop_rect(image, rect):
    """Recorte alineado al eje que contiene el rectangulo rotado `rect`."""
    x, y, w, h = cv2.boundingRect(np.round(cv2.boxPoints(rect)).astype(np.int32))
    x, y = max(x, 0), max(y, 0)
    return image[y:y + h, x:x + w]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", "--limit", type=int, default=None,
                        help="procesar solo las N primeras imagenes de cada vista")
    args = parser.parse_args()

    out_root = config.OUT_DIR / "detections"
    out_root.mkdir(parents=True, exist_ok=True)

    for view in config.VIEWS:
        directory = config.REAL_PLATES_DIR / view
        if not directory.is_dir():
            continue
        out_dir = out_root / view
        out_dir.mkdir(exist_ok=True)

        paths, _ = unique_images(directory)
        if args.limit:
            paths = paths[:args.limit]

        crops, n_detected = [], 0
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                continue

            candidates = detectPlates(image)
            angles = [normalized_angle(cv2.minAreaRect(c)) for c in candidates]
            annotated = draw_candidates(image, candidates, angles)

            scale = PREVIEW_WIDTH / annotated.shape[1]
            annotated = cv2.resize(annotated, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(out_dir / path.name), annotated)

            if candidates:
                n_detected += 1
                crop = crop_rect(image, cv2.minAreaRect(candidates[0]))
                if crop.size:
                    crops.append(cv2.resize(crop, CROP_SIZE))

        print(f"{view}: {n_detected}/{len(paths)} con candidato -> {out_dir}")
        if crops:
            mosaic_path = out_root / f"mosaico_{view}.jpg"
            cv2.imwrite(str(mosaic_path), np.vstack(crops))
            print(f"         mosaico de recortes -> {mosaic_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
