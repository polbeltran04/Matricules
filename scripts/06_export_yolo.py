"""Exporta las detecciones como pre-anotaciones en formato YOLO.

Anotar 202 cajas a mano es caro. Como ya sabemos que el detector morfologico
acierta en 134 de 200 (`validation/detection_verdicts.csv`), se aprovecha ese
trabajo: para esas se escribe la caja que encontro, y las demas quedan
marcadas como pendientes de dibujar.

Eso convierte la tarea en "revisar y corregir" en vez de "dibujar todo", que es
la practica habitual de etiquetado asistido por modelo.

  ATENCION: una pre-anotacion no es una anotacion. La caja del morfologico
  contiene la matricula pero suele venir holgada o cortada, y una caja mal
  ajustada ensena mal a YOLO. Hay que repasarlas todas en un editor
  (labelImg, CVAT, Roboflow, Label Studio) antes de entrenar.

Salida (estructura estandar de YOLO):

    yolo_dataset/
      images/{train,val}/<matricula>.jpg     enlaces o copias
      labels/{train,val}/<matricula>.txt     una linea: clase cx cy w h
      data.yaml                              config para ultralytics
      PENDIENTES.txt                         las que hay que dibujar a mano

Uso:
    python scripts/06_export_yolo.py               # solo escribe las etiquetas
    python scripts/06_export_yolo.py --copy-images # copia tambien las fotos
    python scripts/06_export_yolo.py --val 0.25    # proporcion de validacion
"""

import argparse
import csv
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from alpr.detection import detectPlates

VERDICTS = config.ROOT / "validation" / "detection_verdicts.csv"
OUT_DIR = config.ROOT / "yolo_dataset"
SEED = 42


def yolo_line(rect, width, height, cls=0):
    """minAreaRect -> linea YOLO (caja recta normalizada)."""
    x, y, w, h = cv2.boundingRect(np.round(cv2.boxPoints(rect)).astype(np.int32))
    cx, cy = (x + w / 2) / width, (y + h / 2) / height
    nw, nh = w / width, h / height
    # Recortar a [0,1]: la caja puede salirse del borde de la imagen.
    cx, cy = min(max(cx, 0), 1), min(max(cy, 0), 1)
    nw, nh = min(nw, 1), min(nh, 1)
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def source_path(row):
    root = config.REAL_PLATES_DIR if row["source"] == "real_plates" else config.NEW_PLATES_DIR
    return root / row["view"] / row["file"]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--val", type=float, default=0.2,
                        help="proporcion para validacion (por defecto 0.2)")
    parser.add_argument("--copy-images", action="store_true",
                        help="copiar las fotos al dataset (ocupa espacio)")
    parser.add_argument("--force", action="store_true",
                        help="sobrescribir etiquetas ya existentes (PIERDE las "
                             "correcciones hechas a mano)")
    args = parser.parse_args()

    if not VERDICTS.exists():
        print(f"ERROR: no encuentro {VERDICTS}")
        return 1

    with open(VERDICTS, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    # Split estratificado por dataset y vista, para que train y val tengan la
    # misma mezcla de condiciones. Semilla fija: el split debe ser reproducible.
    rng = random.Random(SEED)
    splits = {}
    groups = {}
    for row in rows:
        groups.setdefault((row["source"], row["view"]), []).append(row)
    for key, group in groups.items():
        group = sorted(group, key=lambda r: r["file"])
        rng.shuffle(group)
        cut = round(len(group) * args.val)
        for i, row in enumerate(group):
            splits[id(row)] = "val" if i < cut else "train"

    for split in ("train", "val"):
        (OUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    written, pending, missing, kept = 0, [], 0, 0
    for row in rows:
        path = source_path(row)
        if not path.exists():
            missing += 1
            continue
        split = splits[id(row)]
        stem = f"{row['source'][0]}_{row['view'][0]}_{path.stem}"
        label_path = OUT_DIR / "labels" / split / f"{stem}.txt"

        # Nunca pisar una etiqueta ya corregida a mano: es trabajo irrecuperable.
        if label_path.exists() and label_path.stat().st_size > 0 and not args.force:
            kept += 1
            continue

        if row["hit"] == "yes":
            image = cv2.imread(str(path))
            candidates = detectPlates(image)
            if candidates:
                rect = cv2.minAreaRect(candidates[0])
                label_path.write_text(
                    yolo_line(rect, image.shape[1], image.shape[0]) + "\n",
                    encoding="utf-8")
                written += 1
            else:
                label_path.write_text("", encoding="utf-8")
                pending.append(f"{split}/{stem}")
        else:
            # Sin caja fiable: fichero vacio y a la lista de pendientes.
            label_path.write_text("", encoding="utf-8")
            pending.append(f"{split}/{stem}")

        if args.copy_images:
            shutil.copy2(path, OUT_DIR / "images" / split / f"{stem}.jpg")

    (OUT_DIR / "data.yaml").write_text(
        f"path: {OUT_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "nc: 1\n"
        "names: [plate]\n", encoding="utf-8")

    (OUT_DIR / "PENDIENTES.txt").write_text(
        "Imagenes sin caja fiable: hay que dibujarlas a mano.\n"
        "Las que si tienen caja son PRE-anotaciones del detector morfologico:\n"
        "repasalas igualmente, suelen venir holgadas o cortadas.\n\n"
        + "\n".join(sorted(pending)) + "\n", encoding="utf-8")

    n_train = len(list((OUT_DIR / "labels" / "train").glob("*.txt")))
    n_val = len(list((OUT_DIR / "labels" / "val").glob("*.txt")))
    print(f"dataset YOLO -> {OUT_DIR}")
    print(f"  train {n_train}  |  val {n_val}")
    print(f"  {written} cajas pre-anotadas ({written / max(len(rows), 1):.0%})")
    print(f"  {len(pending)} pendientes de dibujar a mano -> PENDIENTES.txt")
    if kept:
        print(f"  {kept} etiquetas ya existentes, respetadas (--force para rehacerlas)")
    if missing:
        print(f"  {missing} imagenes del CSV no estan en disco")
    if not args.copy_images:
        print("\n  (las fotos no se han copiado; usa --copy-images si las quieres dentro)")
    print("\n  Siguiente paso: abrir el dataset en un editor de anotaciones")
    print("  (labelImg, CVAT, Roboflow...) y repasar TODAS las cajas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
