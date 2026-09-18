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
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from alpr.dataset import draw_candidates, plate_from_filename, unique_images
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
    parser.add_argument("--new", action="store_true",
                        help="incluir tambien new_plates/")
    args = parser.parse_args()

    out_root = config.OUT_DIR / "detections"
    out_root.mkdir(parents=True, exist_ok=True)

    roots = [config.REAL_PLATES_DIR]
    if args.new and config.NEW_PLATES_DIR.is_dir():
        roots.append(config.NEW_PLATES_DIR)

    index = []
    for root in roots:
        for view in config.VIEWS:
            directory = root / view
            if not directory.is_dir():
                continue
            out_dir = out_root / root.name / view
            crop_dir = out_root / "crops" / root.name / view
            out_dir.mkdir(parents=True, exist_ok=True)
            crop_dir.mkdir(parents=True, exist_ok=True)

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
                cv2.imwrite(str(out_dir / path.name),
                            cv2.resize(annotated, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA))

                row = {"source": root.name, "view": view, "file": path.name,
                       "plate": plate_from_filename(path) or "",
                       "n_candidates": len(candidates),
                       "angle": round(angles[0], 2) if angles else "",
                       "crop": ""}
                if candidates:
                    n_detected += 1
                    crop = crop_rect(image, cv2.minAreaRect(candidates[0]))
                    if crop.size:
                        # Un fichero por imagen, nombrado con la matricula: al
                        # mirarlo se compara lo que pone la placa con el nombre,
                        # y eso da un veredicto de acierto sin anotar cajas.
                        crop_path = crop_dir / path.name
                        cv2.imwrite(str(crop_path), cv2.resize(crop, CROP_SIZE))
                        crops.append(cv2.resize(crop, CROP_SIZE))
                        row["crop"] = str(crop_path.relative_to(config.OUT_DIR))
                index.append(row)

            print(f"{root.name}/{view}: {n_detected}/{len(paths)} con candidato")
            print(f"    anotadas -> {out_dir}")
            print(f"    recortes -> {crop_dir}")
            if crops:
                mosaic = out_root / f"mosaico_{root.name}_{view}.jpg"
                cv2.imwrite(str(mosaic), np.vstack(crops))
                print(f"    mosaico  -> {mosaic}")

    # Indice para la revision: una fila por imagen, con su recorte.
    index_path = out_root / "candidatos.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["source", "view", "file", "plate",
                                                "n_candidates", "angle", "crop"])
        writer.writeheader()
        writer.writerows(index)
    print(f"\nindice de candidatos -> {index_path} ({len(index)} filas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
