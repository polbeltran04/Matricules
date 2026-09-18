"""Revision interactiva de cajas: aprobar la propuesta o dibujarla a mano.

Muestra cada imagen con la caja que propuso el detector y espera un veredicto.
Lo aprobado o dibujado se guarda en formato YOLO, y la imagen queda marcada
como revisada para no volver a salir.

Por eso se puede parar y seguir cuando sea, y al anadir fotos nuevas solo
pedira las que falten.

CONTROLES
    a / ESPACIO   aprobar la caja propuesta
    d             dibujarla a mano (arrastrar con el raton)
    n             la imagen no tiene matricula visible (etiqueta vacia)
    s             saltar por ahora (volvera a salir)
    z             deshacer la ultima decision
    q / ESC       guardar y salir

  En modo dibujo: arrastrar con el boton izquierdo, ENTER confirma, r rehace,
  ESC vuelve sin cambios.

Uso:
    python scripts/07_annotate.py                 # todas las no revisadas
    python scripts/07_annotate.py --only-pending  # solo las que no tienen caja
    python scripts/07_annotate.py --source new_plates
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

import config

YOLO_DIR = config.ROOT / "yolo_dataset"
REVIEWED = YOLO_DIR / "reviewed.csv"
WINDOW = "Anotador de matriculas"
VIEW_WIDTH = 1200
GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 255, 255), (255, 255, 255)


def load_reviewed():
    if not REVIEWED.exists():
        return {}
    with open(REVIEWED, newline="", encoding="utf-8") as fh:
        return {r["stem"]: r for r in csv.DictReader(fh)}


def save_reviewed(entries):
    REVIEWED.parent.mkdir(parents=True, exist_ok=True)
    with open(REVIEWED, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["stem", "split", "status", "when"])
        writer.writeheader()
        writer.writerows(sorted(entries.values(), key=lambda r: r["stem"]))


def find_source_image(stem):
    """`n_F_1234ABC` -> la foto original en real_plates/ o new_plates/."""
    tag_source, tag_view, name = stem.split("_", 2)
    root = config.REAL_PLATES_DIR if tag_source == "r" else config.NEW_PLATES_DIR
    for view in config.VIEWS:
        if view[0] == tag_view:
            path = root / view / f"{name}.jpg"
            if path.exists():
                return path
    return None


def read_box(label_path):
    """Lee la caja YOLO del .txt, o None si no hay."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return None
    parts = label_path.read_text(encoding="utf-8").split()
    if len(parts) < 5:
        return None
    return tuple(float(v) for v in parts[1:5])


def write_box(label_path, box):
    """Escribe la caja YOLO, o un fichero vacio si box es None."""
    text = "" if box is None else "0 " + " ".join(f"{v:.6f}" for v in box) + "\n"
    label_path.write_text(text, encoding="utf-8")


def box_to_pixels(box, width, height):
    cx, cy, w, h = box
    return (int((cx - w / 2) * width), int((cy - h / 2) * height),
            int(w * width), int(h * height))


def pixels_to_box(x, y, w, h, width, height):
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


class Annotator:
    def __init__(self, items):
        self.items = items
        self.i = 0
        self.history = []
        self.drawing = False
        self.start = None
        self.temp = None
        self.mode = "review"

    def on_mouse(self, event, x, y, flags, _param):
        if self.mode != "draw":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.start, self.temp = True, (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp = (*self.start, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp = (*self.start, x, y)

    def render(self, frame, item, box_px):
        canvas = frame.copy()
        h = canvas.shape[0]
        if self.mode == "draw":
            if self.temp:
                x0, y0, x1, y1 = self.temp
                cv2.rectangle(canvas, (x0, y0), (x1, y1), YELLOW, 2)
            help_text = "DIBUJA: arrastra | ENTER confirma | r rehace | ESC cancela"
            colour = YELLOW
        else:
            if box_px:
                x, y, w, h_box = box_px
                cv2.rectangle(canvas, (x, y), (x + w, y + h_box), GREEN, 2)
                help_text = "a aprobar | d dibujar | n sin matricula | s saltar | z deshacer | q salir"
                colour = GREEN
            else:
                help_text = "SIN CAJA: d dibujar | n sin matricula | s saltar | z deshacer | q salir"
                colour = RED

        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
        cv2.putText(canvas, f"[{self.i + 1}/{len(self.items)}]  {item['plate']}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, h - 26), (canvas.shape[1], h), (0, 0, 0), -1)
        cv2.putText(canvas, f"{item['source']}/{item['view']}/{item['stem']}",
                    (10, h - 8), cv2.FONT_HERSHEY_SIMPLEX, .45, (170, 170, 170), 1, cv2.LINE_AA)
        return canvas

    def run(self, reviewed):
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        while 0 <= self.i < len(self.items):
            item = self.items[self.i]
            image = cv2.imread(str(item["path"]))
            if image is None:
                self.i += 1
                continue

            scale = VIEW_WIDTH / image.shape[1]
            frame = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            fh, fw = frame.shape[:2]
            box = read_box(item["label"])
            box_px = box_to_pixels(box, fw, fh) if box else None
            self.mode, self.temp = "review", None

            decided = False
            while not decided:
                cv2.imshow(WINDOW, self.render(frame, item, box_px))
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return
                    continue

                if self.mode == "draw":
                    if key == 13 and self.temp:                     # ENTER
                        x0, y0, x1, y1 = self.temp
                        x, y = min(x0, x1), min(y0, y1)
                        w, h = abs(x1 - x0), abs(y1 - y0)
                        if w > 5 and h > 5:
                            write_box(item["label"], pixels_to_box(x, y, w, h, fw, fh))
                            self.record(reviewed, item, "drawn")
                            decided = True
                    elif key in (ord("r"), ord("R")):
                        self.temp = None
                    elif key == 27:                                  # ESC
                        self.mode, self.temp = "review", None
                    continue

                if key in (ord("a"), ord("A"), 32) and box:          # aprobar
                    self.record(reviewed, item, "approved")
                    decided = True
                elif key in (ord("d"), ord("D")):
                    self.mode, self.temp = "draw", None
                elif key in (ord("n"), ord("N")):
                    write_box(item["label"], None)
                    self.record(reviewed, item, "no_plate")
                    decided = True
                elif key in (ord("s"), ord("S")):
                    self.i += 1
                    decided = True
                elif key in (ord("z"), ord("Z")):
                    if self.history:
                        prev = self.history.pop()
                        reviewed.pop(prev["stem"], None)
                        self.i = prev["index"]
                    decided = True
                elif key in (ord("q"), ord("Q"), 27):
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()

    def record(self, reviewed, item, status):
        self.history.append({"stem": item["stem"], "index": self.i})
        reviewed[item["stem"]] = {"stem": item["stem"], "split": item["split"],
                                  "status": status,
                                  "when": datetime.now().isoformat(timespec="seconds")}
        save_reviewed(reviewed)
        self.i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-pending", action="store_true",
                        help="solo las imagenes que aun no tienen ninguna caja")
    parser.add_argument("--source", choices=["real_plates", "new_plates"],
                        help="limitar a un dataset")
    parser.add_argument("--redo", action="store_true",
                        help="incluir tambien las ya revisadas")
    parser.add_argument("--list", action="store_true",
                        help="listar lo que queda por revisar y salir, sin abrir ventana")
    args = parser.parse_args()

    if not YOLO_DIR.is_dir():
        print(f"ERROR: no encuentro {YOLO_DIR}. Ejecuta antes scripts/06_export_yolo.py")
        return 1

    reviewed = load_reviewed()
    items = []
    for split in ("train", "val"):
        for label in sorted((YOLO_DIR / "labels" / split).glob("*.txt")):
            stem = label.stem
            if stem in reviewed and not args.redo:
                continue
            has_box = read_box(label) is not None
            if args.only_pending and has_box:
                continue
            path = find_source_image(stem)
            if path is None:
                continue
            source = "real_plates" if stem.startswith("r_") else "new_plates"
            if args.source and source != args.source:
                continue
            items.append({"stem": stem, "split": split, "label": label, "path": path,
                          "source": source, "view": path.parent.name,
                          "plate": path.stem})

    if not items:
        print("No queda nada por revisar.")
        print(f"  {len(reviewed)} imagenes ya revisadas -> {REVIEWED}")
        return 0

    with_box = sum(1 for it in items if read_box(it["label"]))
    print(f"{len(items)} imagenes por revisar ({with_box} con caja propuesta, "
          f"{len(items) - with_box} sin ella)")
    print(f"Ya revisadas: {len(reviewed)}\n")

    if args.list:
        for it in items[:15]:
            mark = "caja" if read_box(it["label"]) else "----"
            print(f"  [{mark}] {it['split']:<5} {it['source']:<11} {it['view']:<7} {it['plate']}")
        if len(items) > 15:
            print(f"  ... y {len(items) - 15} mas")
        return 0

    print("  a/ESPACIO aprobar | d dibujar | n sin matricula | s saltar | z deshacer | q salir\n")

    Annotator(items).run(reviewed)

    counts = {}
    for row in reviewed.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print(f"\nrevisadas en total: {len(reviewed)} -> {REVIEWED}")
    for status, n in sorted(counts.items()):
        print(f"  {status:<10} {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
