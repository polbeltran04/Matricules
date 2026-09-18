"""Mide el IoU del detector morfologico contra las cajas anotadas a mano.

Esta es la respuesta a "como validamos que detectamos bien la matricula".
Hasta ahora habia dos numeros, y ninguno medía lo ajustada que esta la caja:

    cobertura 99.0%   -> en cuantas imagenes devolvio ALGO (un faro cuenta)
    acierto   67.0%   -> en cuantas el recorte contenia la matricula legible

El IoU es el tercero y el unico que mide la localizacion: cuanto se solapa la
caja propuesta con la real. Con las 200 imagenes anotadas ya se puede calcular.

Se reporta a dos umbrales:

    IoU >= 0.5   el estandar (PASCAL VOC, el mAP@0.5 que publica YOLO)
    IoU >= 0.7   el que conviene para ALPR: detras viene segmentar caracteres,
                 y una caja que corta la placa arruina esa etapa aunque
                 solape medio.

Ademas, para las que tienen cuadrilatero se mide la CONTENCION: que fraccion de
la matricula real cae dentro de la caja propuesta. Un IoU bajo con contencion
1.0 es una caja holgada (recuperable, solo sobra fondo); un IoU alto con
contencion 0.9 es una caja que corta (no recuperable, se pierden caracteres).
Distinguirlos importa mas que el IoU medio.

Uso:
    python scripts/08_iou_baseline.py
    python scripts/08_iou_baseline.py --csv    # volcar out/iou_baseline.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config
from alpr.detection import detectPlates

YOLO_DIR = config.ROOT / "yolo_dataset"
REVIEWED = YOLO_DIR / "reviewed.csv"
QUADS = YOLO_DIR / "quads.csv"


def load_reviewed():
    with open(REVIEWED, newline="", encoding="utf-8") as fh:
        return {r["stem"]: r for r in csv.DictReader(fh)}


def load_quads():
    if not QUADS.exists():
        return {}
    out = {}
    with open(QUADS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["stem"]] = [(float(row[f"x{i}"]), float(row[f"y{i}"]))
                                for i in range(1, 5)]
    return out


def find_source_image(stem):
    tag_source, tag_view, name = stem.split("_", 2)
    root = config.REAL_PLATES_DIR if tag_source == "r" else config.NEW_PLATES_DIR
    for view in config.VIEWS:
        if view[0] == tag_view:
            path = root / view / f"{name}.jpg"
            if path.exists():
                return path
    return None


def read_box(label_path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return None
    parts = label_path.read_text(encoding="utf-8").split()
    return tuple(float(v) for v in parts[1:5]) if len(parts) >= 5 else None


def yolo_to_xyxy(box, width, height):
    cx, cy, w, h = box
    return ((cx - w / 2) * width, (cy - h / 2) * height,
            (cx + w / 2) * width, (cy + h / 2) * height)


def iou(a, b):
    """IoU de dos cajas (x0, y0, x1, y1)."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def containment(quad_px, box):
    """Fraccion del cuadrilatero de la placa que cae dentro de la caja.
    1.0 = la matricula entera esta dentro; < 1.0 = la caja la corta."""
    x0, y0, x1, y1 = box
    # Recortar el poligono contra el rectangulo con Sutherland-Hodgman.
    poly = list(quad_px)
    for inside, intersect in (
            (lambda p: p[0] >= x0, lambda p, q: (x0, p[1] + (q[1] - p[1]) * (x0 - p[0]) / (q[0] - p[0]))),
            (lambda p: p[0] <= x1, lambda p, q: (x1, p[1] + (q[1] - p[1]) * (x1 - p[0]) / (q[0] - p[0]))),
            (lambda p: p[1] >= y0, lambda p, q: (p[0] + (q[0] - p[0]) * (y0 - p[1]) / (q[1] - p[1]), y0)),
            (lambda p: p[1] <= y1, lambda p, q: (p[0] + (q[0] - p[0]) * (y1 - p[1]) / (q[1] - p[1]), y1))):
        if not poly:
            break
        out, prev = [], poly[-1]
        for point in poly:
            if inside(point):
                if not inside(prev):
                    out.append(intersect(prev, point))
                out.append(point)
            elif inside(prev):
                out.append(intersect(prev, point))
            prev = point
        poly = out
    if len(poly) < 3:
        return 0.0
    total = cv2.contourArea(np.array(quad_px, np.float32))
    return cv2.contourArea(np.array(poly, np.float32)) / total if total else 0.0


def summarise(name, values, thresholds=(0.5, 0.7)):
    if not values:
        return
    arr = np.array(values)
    linea = f"  {name:<24} n={len(arr):<4} IoU medio {arr.mean():.3f}  mediana {np.median(arr):.3f}"
    for t in thresholds:
        linea += f"   >={t}: {(arr >= t).mean():6.1%}"
    print(linea)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", action="store_true",
                        help="volcar el detalle en out/iou_baseline.csv")
    args = parser.parse_args()

    if not REVIEWED.exists():
        print(f"ERROR: no encuentro {REVIEWED}. Anota antes con scripts/07_annotate.py")
        return 1

    reviewed = load_reviewed()
    quads = load_quads()
    rows = []

    for stem, entry in sorted(reviewed.items()):
        label = YOLO_DIR / "labels" / entry["split"] / f"{stem}.txt"
        truth = read_box(label)
        path = find_source_image(stem)
        if truth is None or path is None:
            continue

        image = cv2.imread(str(path))
        if image is None:
            continue
        height, width = image.shape[:2]
        gt = yolo_to_xyxy(truth, width, height)

        # El detector, tal cual lo usa el resto del proyecto.
        candidates = detectPlates(image)
        if candidates:
            x, y, w, h = cv2.boundingRect(
                np.round(cv2.boxPoints(cv2.minAreaRect(candidates[0]))).astype(np.int32))
            pred = (x, y, x + w, y + h)
            score = iou(pred, gt)
        else:
            pred, score = None, 0.0

        # Contencion de la placa real: solo se puede medir donde hay
        # cuadrilatero, que es la unica anotacion exacta de la matricula.
        cover = ""
        if pred is not None and stem in quads:
            quad_px = [(x * width, y * height) for x, y in quads[stem]]
            cover = containment(quad_px, pred)

        rows.append({"stem": stem, "split": entry["split"], "status": entry["status"],
                     "source": "real_plates" if stem.startswith("r_") else "new_plates",
                     "view": path.parent.name, "plate": path.stem,
                     "detected": int(pred is not None),
                     "iou": round(score, 4),
                     "containment": round(cover, 4) if cover != "" else ""})

    if not rows:
        print("No hay nada que medir.")
        return 1

    ious = [r["iou"] for r in rows]
    print(f"\nIoU del detector morfologico contra {len(rows)} cajas anotadas a mano\n")
    summarise("TOTAL", ious)
    for source in ("real_plates", "new_plates"):
        summarise(source, [r["iou"] for r in rows if r["source"] == source])
    for view in config.VIEWS:
        summarise(f"  vista {view}", [r["iou"] for r in rows if r["view"] == view])

    sin_deteccion = sum(1 for r in rows if not r["detected"])
    if sin_deteccion:
        print(f"\n  {sin_deteccion} imagenes sin ningun candidato (IoU 0 por defecto)")

    # La contencion solo es interpretable cuando el detector esta SOBRE la placa.
    # Si cogio un faro, la contencion es 0 pero el problema no es que corte:
    # es que no la ha encontrado. Son dos fallos distintos y se cuentan aparte.
    con_quad = [r for r in rows if r["containment"] != ""]
    if con_quad:
        localizadas = [r["containment"] for r in con_quad if r["iou"] >= 0.5]
        perdidas = [r for r in con_quad if r["iou"] < 0.5]
        print(f"\nContencion de la matricula real (n={len(con_quad)}, las que "
              f"tienen cuadrilatero)")
        print(f"  no la localiza siquiera (IoU<0.5): {len(perdidas):3d}  "
              f"{len(perdidas)/len(con_quad):6.1%}")
        if localizadas:
            arr = np.array(localizadas)
            print(f"  de las {len(arr)} que SI localiza:")
            print(f"     la coge entera (>=0.99): {(arr >= 0.99).mean():6.1%}")
            print(f"     la corta        (<0.99): {(arr < 0.99).mean():6.1%}"
                  f"   (contencion mediana {np.median(arr):.3f})")
        print("\n  Una caja holgada solo sobra fondo y se recorta despues.")
        print("  Una caja que corta pierde caracteres: eso no se recupera.")

    if args.csv:
        config.OUT_DIR.mkdir(parents=True, exist_ok=True)
        out = config.OUT_DIR / "iou_baseline.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\ndetalle -> {out}")

    print("\n  Esta es la linea base. YOLO se compara contra estos mismos numeros,")
    print("  sobre el mismo split, que es el Obj3 del enunciado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
