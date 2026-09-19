"""Revision interactiva de cajas: aprobar la propuesta, marcar esquinas o dibujar.

Muestra cada imagen con la caja que propuso el detector y espera un veredicto.
Lo aprobado o marcado se guarda en formato YOLO, y la imagen queda marcada
como revisada para no volver a salir.

Por eso se puede parar y seguir cuando sea, y al anadir fotos nuevas solo
pedira las que falten.

DOS FORMAS DE MARCAR

  Esquinas (c) - RECOMENDADO. Se hace clic en las 4 esquinas de la matricula.
      Una caja recta sobre una placa inclinada mete parachoques por fuerza; el
      cuadrilatero no. De el se deriva la caja YOLO (que sigue siendo recta,
      es lo que pide el formato) anadiendo un margen FIJO e igual para todas,
      asi que el margen deja de depender del pulso.
      Ademas el cuadrilatero es lo que hace falta en la Sesion 2 para
      rectificar la perspectiva antes de segmentar caracteres.

  Rectangulo (d) - arrastrar. Rapido, pero recoge fondo si la placa va torcida.

CONTROLES
    a / ESPACIO   aprobar la caja propuesta
    c             marcar las 4 esquinas de la matricula
    d             dibujar un rectangulo arrastrando
    n             la imagen no tiene matricula visible (etiqueta vacia)
    s             saltar por ahora (volvera a salir)
    z             deshacer la ultima decision
    + / -         acercar / alejar la imagen
    q / ESC       guardar y salir

  En esquinas: clic en las 4 (cualquier orden), RETROCESO quita la ultima,
  r empieza de cero, ENTER confirma, ESC vuelve sin cambios.
  En rectangulo: arrastrar con el boton izquierdo, ENTER confirma, r rehace,
  ESC vuelve sin cambios.

  La lupa sigue al raton y muestra los pixeles a tamano real, para colocar las
  esquinas con precision aunque la foto se vea reducida.

Uso:
    python scripts/07_annotate.py                 # todas las no revisadas
    python scripts/07_annotate.py --only-pending  # solo las que no tienen caja
    python scripts/07_annotate.py --review        # repasar las YA guardadas
    python scripts/07_annotate.py --source new_plates
    python scripts/07_annotate.py --rebuild       # rehacer las cajas desde los
                                                  # cuadrilateros con otro margen

REPASO (--review)
    Vuelve a mostrar las cajas ya guardadas, sobre la imagen entera, para
    validarlas o rehacerlas. Van ordenadas de mas a menos sospechosa:

      1. las marcadas "sin matricula", que son las de decision mas dudosa;
      2. las de caja mas achatada (AR bajo), que es el sintoma de haber
         recogido parachoques al arrastrar sobre una placa inclinada;
      3. el resto.

    Asi las ruidosas salen al principio en vez de haber que buscarlas.
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
QUADS = YOLO_DIR / "quads.csv"
WINDOW = "Anotador de matriculas"

# Margen que se anade a la caja YOLO derivada del cuadrilatero, en fraccion del
# lado. Un poco de holgura no penaliza el IoU y protege a la etapa siguiente;
# cortar un caracter si la arruina. Ver --pad para cambiarlo.
PAD_FRAC = 0.04

SCREEN_MARGIN = 150        # hueco para barra de tareas y marco de ventana
LOUPE, LOUPE_ZOOM = 200, 4
GREEN, RED, YELLOW, WHITE, CYAN = ((0, 220, 0), (0, 0, 255), (0, 255, 255),
                                   (255, 255, 255), (255, 200, 0))
QUAD_FIELDS = ["stem", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def screen_size():
    """Resolucion de pantalla, para que la foto quepa entera."""
    try:
        import tkinter
        root = tkinter.Tk()
        root.withdraw()
        size = (root.winfo_screenwidth(), root.winfo_screenheight())
        root.destroy()
        return size
    except Exception:
        return (1280, 720)


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


def load_quads():
    """stem -> [(x,y) x4] normalizado. Es la anotacion precisa de la placa."""
    if not QUADS.exists():
        return {}
    out = {}
    with open(QUADS, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["stem"]] = [(float(row[f"x{i}"]), float(row[f"y{i}"]))
                                for i in range(1, 5)]
    return out


def save_quads(quads):
    QUADS.parent.mkdir(parents=True, exist_ok=True)
    with open(QUADS, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=QUAD_FIELDS)
        writer.writeheader()
        for stem in sorted(quads):
            row = {"stem": stem}
            for i, (x, y) in enumerate(quads[stem], start=1):
                row[f"x{i}"], row[f"y{i}"] = f"{x:.6f}", f"{y:.6f}"
            writer.writerow(row)


def order_quad(points):
    """Ordena 4 puntos a TL, TR, BR, BL. Asi el cuadrilatero sirve tal cual
    para `cv2.getPerspectiveTransform` en la Sesion 2."""
    pts = np.array(points, dtype=np.float64)
    s, d = pts.sum(axis=1), np.diff(pts, axis=1).ravel()
    return [tuple(pts[np.argmin(s)]), tuple(pts[np.argmin(d)]),
            tuple(pts[np.argmax(s)]), tuple(pts[np.argmax(d)])]


def box_from_quad(quad, pad=PAD_FRAC):
    """Cuadrilatero -> caja YOLO recta, con margen fijo y recortada a [0,1]."""
    xs, ys = [p[0] for p in quad], [p[1] for p in quad]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x1 - x0) * pad, (y1 - y0) * pad
    x0, x1 = max(x0 - mx, 0.0), min(x1 + mx, 1.0)
    y0, y1 = max(y0 - my, 0.0), min(y1 + my, 1.0)
    return ((x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0)


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


def image_size(path):
    """(ancho, alto) leyendo solo la cabecera del JPEG, sin decodificarlo."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        image = cv2.imread(str(path))
        return (0, 0) if image is None else (image.shape[1], image.shape[0])


def box_aspect(box, width, height):
    """AR de la caja en pixeles. En YOLO w y h son fracciones de lados
    distintos, asi que w/h NO es el AR real hasta multiplicar por la imagen."""
    if not box or not width or not height:
        return None
    _, _, w, h = box
    return (w * width) / (h * height) if h else None


def label_path_for(stem):
    for split in ("train", "val"):
        path = YOLO_DIR / "labels" / split / f"{stem}.txt"
        if path.exists():
            return path, split
    return None, None


def box_to_pixels(box, width, height):
    cx, cy, w, h = box
    return (int((cx - w / 2) * width), int((cy - h / 2) * height),
            int(w * width), int(h * height))


def pixels_to_box(x, y, w, h, width, height):
    return ((x + w / 2) / width, (y + h / 2) / height, w / width, h / height)


class Annotator:
    def __init__(self, items, quads, pad, review=False):
        self.items = items
        self.quads = quads
        self.pad = pad
        self.review = review
        self.i = 0
        self.history = []
        self.drawing = False
        self.start = None
        self.temp = None
        self.points = []
        self.mouse = None
        self.mode = "review"
        self.zoom = 1.0
        self.skipped = set()

    def on_mouse(self, event, x, y, flags, _param):
        self.mouse = (x, y)
        if self.mode == "quad":
            if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
                self.points.append((x, y))
            return
        if self.mode != "draw":
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing, self.start, self.temp = True, (x, y), None
        elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
            self.temp = (*self.start, x, y)
        elif event == cv2.EVENT_LBUTTONUP and self.drawing:
            self.drawing = False
            self.temp = (*self.start, x, y)

    def draw_loupe(self, canvas, image, scale):
        """Recuadro con los pixeles originales alrededor del raton: sin el, en
        una foto de 4000 px reducida a 1200 no se ve donde cae la esquina."""
        if self.mouse is None or self.mode == "review":
            return
        ch, cw = canvas.shape[:2]
        mx, my = self.mouse
        if not (0 <= mx < cw and 0 <= my < ch):
            return
        half = LOUPE // (2 * LOUPE_ZOOM)
        ox, oy = int(mx / scale), int(my / scale)
        patch = cv2.copyMakeBorder(image, half, half, half, half,
                                   cv2.BORDER_CONSTANT, value=(30, 30, 30))
        patch = patch[oy:oy + 2 * half, ox:ox + 2 * half]
        if patch.shape[0] != 2 * half or patch.shape[1] != 2 * half:
            return
        zoom = cv2.resize(patch, (LOUPE, LOUPE), interpolation=cv2.INTER_NEAREST)
        cv2.line(zoom, (LOUPE // 2, 0), (LOUPE // 2, LOUPE), CYAN, 1)
        cv2.line(zoom, (0, LOUPE // 2), (LOUPE, LOUPE // 2), CYAN, 1)
        cv2.rectangle(zoom, (0, 0), (LOUPE - 1, LOUPE - 1), WHITE, 1)
        # Al lado contrario del raton, para no tapar lo que se esta marcando.
        px = 10 if mx > cw // 2 else cw - LOUPE - 10
        py = 70
        if py + LOUPE <= ch and px >= 0 and px + LOUPE <= cw:
            canvas[py:py + LOUPE, px:px + LOUPE] = zoom

    def render(self, frame, image, scale, item, box_px):
        canvas = frame.copy()
        ch, cw = canvas.shape[:2]

        if self.mode == "quad":
            for n, (px, py) in enumerate(self.points, start=1):
                cv2.circle(canvas, (px, py), 5, CYAN, -1)
                cv2.putText(canvas, str(n), (px + 8, py - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, CYAN, 2, cv2.LINE_AA)
            if len(self.points) >= 2:
                closed = len(self.points) == 4
                cv2.polylines(canvas, [np.array(self.points, np.int32)],
                              closed, CYAN, 2)
            if len(self.points) == 4:
                quad = order_quad([(p[0] / cw, p[1] / ch) for p in self.points])
                x, y, w, h = box_to_pixels(box_from_quad(quad, self.pad), cw, ch)
                cv2.rectangle(canvas, (x, y), (x + w, y + h), GREEN, 1)
                help_text = ("4/4 esquinas | ENTER confirma | RETROCESO quita la "
                             "ultima | r de cero | ESC cancela")
            else:
                help_text = (f"ESQUINAS {len(self.points)}/4: clic en cada esquina "
                             "| RETROCESO quita | r de cero | ESC cancela")
            colour = CYAN
        elif self.mode == "draw":
            if self.temp:
                x0, y0, x1, y1 = self.temp
                cv2.rectangle(canvas, (x0, y0), (x1, y1), YELLOW, 2)
            help_text = "DIBUJA: arrastra | ENTER confirma | r rehace | ESC cancela"
            colour = YELLOW
        else:
            quad = self.quads.get(item["stem"])
            if quad:
                pts = np.array([(int(x * cw), int(y * ch)) for x, y in quad], np.int32)
                cv2.polylines(canvas, [pts], True, CYAN, 2)
            if box_px:
                x, y, w, h = box_px
                cv2.rectangle(canvas, (x, y), (x + w, y + h), GREEN, 2)
                help_text = ("a APROBAR (guarda) | c esquinas | d rectangulo | "
                             "n sin matricula | s saltar sin guardar | z | +/- | q")
                colour = GREEN
            else:
                help_text = ("SIN CAJA: c esquinas | d rectangulo | n sin matricula "
                             "| s saltar sin guardar | z | +/- zoom | q salir")
                colour = RED

        self.draw_loupe(canvas, image, scale)

        cv2.rectangle(canvas, (0, 0), (cw, 62), (0, 0, 0), -1)
        cv2.putText(canvas, f"[{self.i + 1}/{len(self.items)}]  {item['plate']}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, .8, colour, 2, cv2.LINE_AA)
        cv2.putText(canvas, help_text, (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX, .46, WHITE, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (0, ch - 26), (cw, ch), (0, 0, 0), -1)
        ar = item.get("ar")
        # El AR delata la caja ruidosa: una placa espanola es 4.73, y cuanto mas
        # baja, mas parachoques se recogio al arrastrar sobre una placa torcida.
        info = "sin caja" if ar is None else f"AR {ar:.2f}"
        marca = "cuadrilatero" if self.quads.get(item["stem"]) else "rectangulo"
        estado = item.get("status")
        cv2.putText(canvas, f"{item['source']}/{item['view']}/{item['stem']}"
                            f"   {info}   {marca}"
                            f"{'   ' + estado if estado else ''}"
                            f"   zoom {self.zoom:.1f}x",
                    (10, ch - 8), cv2.FONT_HERSHEY_SIMPLEX, .45,
                    (170, 170, 170) if ar is None or ar > 2.2 else (120, 190, 255),
                    1, cv2.LINE_AA)
        return canvas

    def confirm_quad(self, reviewed, item, fw, fh):
        quad = order_quad([(p[0] / fw, p[1] / fh) for p in self.points])
        self.quads[item["stem"]] = quad
        save_quads(self.quads)
        write_box(item["label"], box_from_quad(quad, self.pad))
        self.record(reviewed, item, "quad")

    def run(self, reviewed):
        screen_w, screen_h = screen_size()
        max_w, max_h = screen_w - SCREEN_MARGIN, screen_h - SCREEN_MARGIN
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW, self.on_mouse)

        while 0 <= self.i < len(self.items):
            item = self.items[self.i]
            image = cv2.imread(str(item["path"]))
            if image is None:
                self.i += 1
                continue

            ih, iw = image.shape[:2]
            # Ajustar a la pantalla por ANCHO Y ALTO: una foto vertical
            # 3000x4000 escalada solo por ancho se sale por abajo.
            fit = min(max_w / iw, max_h / ih, 1.0)
            self.mode, self.temp, self.points = "review", None, []
            box = read_box(item["label"])

            decided = False
            redraw = True
            while not decided:
                if redraw:
                    scale = fit * self.zoom
                    frame = cv2.resize(image, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
                    fh, fw = frame.shape[:2]
                    box_px = box_to_pixels(box, fw, fh) if box else None
                    redraw = False

                cv2.imshow(WINDOW, self.render(frame, image, scale, item, box_px))
                key = cv2.waitKey(20) & 0xFF
                if key == 255:
                    if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                        return
                    continue

                if key in (ord("+"), ord("=")):
                    self.zoom, redraw = min(self.zoom * 1.25, 4.0), True
                    self.points = []
                    continue
                if key in (ord("-"), ord("_")):
                    self.zoom, redraw = max(self.zoom / 1.25, 0.4), True
                    self.points = []
                    continue

                if self.mode == "quad":
                    if key == 13 and len(self.points) == 4:          # ENTER
                        self.confirm_quad(reviewed, item, fw, fh)
                        decided = True
                    elif key == 8:                                    # RETROCESO
                        if self.points:
                            self.points.pop()
                    elif key in (ord("r"), ord("R")):
                        self.points = []
                    elif key == 27:                                   # ESC
                        self.mode, self.points = "review", []
                    continue

                if self.mode == "draw":
                    if key == 13 and self.temp:                       # ENTER
                        x0, y0, x1, y1 = self.temp
                        x, y = min(x0, x1), min(y0, y1)
                        w, h = abs(x1 - x0), abs(y1 - y0)
                        if w > 5 and h > 5:
                            write_box(item["label"], pixels_to_box(x, y, w, h, fw, fh))
                            self.quads.pop(item["stem"], None)
                            self.record(reviewed, item, "drawn")
                            decided = True
                    elif key in (ord("r"), ord("R")):
                        self.temp = None
                    elif key == 27:                                   # ESC
                        self.mode, self.temp = "review", None
                    continue

                if key in (ord("a"), ord("A"), 32) and box:           # aprobar
                    # Aprobar una pre-anotacion del detector no es lo mismo que
                    # revalidar algo que ya se habia revisado a mano: se guardan
                    # con estado distinto para poder distinguirlas despues.
                    self.record(reviewed, item,
                                "confirmed" if item.get("status") else "approved")
                    decided = True
                elif key in (ord("c"), ord("C")):
                    self.mode, self.points = "quad", []
                elif key in (ord("d"), ord("D")):
                    self.mode, self.temp = "draw", None
                elif key in (ord("n"), ord("N")):
                    write_box(item["label"], None)
                    self.quads.pop(item["stem"], None)
                    self.record(reviewed, item, "no_plate")
                    decided = True
                elif key in (ord("s"), ord("S")):
                    # Saltar NO guarda: la imagen volvera a salir. Se lleva la
                    # cuenta para avisar al final, porque es facil confundirlo
                    # con aprobar y perder el repaso entero.
                    self.skipped.add(item["stem"])
                    self.i += 1
                    decided = True
                elif key in (ord("z"), ord("Z")):
                    if self.history:
                        prev = self.history.pop()
                        if prev["previous"] is None:
                            reviewed.pop(prev["stem"], None)
                        else:
                            reviewed[prev["stem"]] = prev["previous"]
                        save_reviewed(reviewed)
                        self.i = prev["index"]
                    decided = True
                elif key in (ord("q"), ord("Q"), 27):
                    cv2.destroyAllWindows()
                    return

        cv2.destroyAllWindows()

    def record(self, reviewed, item, status):
        # Guardar la fila anterior, no solo el stem: al repasar, deshacer tiene
        # que devolver el estado que habia, no borrar el registro.
        self.history.append({"stem": item["stem"], "index": self.i,
                             "previous": reviewed.get(item["stem"])})
        reviewed[item["stem"]] = {"stem": item["stem"], "split": item["split"],
                                  "status": status,
                                  "when": datetime.now().isoformat(timespec="seconds")}
        save_reviewed(reviewed)
        item["status"] = status
        self.i += 1


def rebuild(pad):
    """Rehace las cajas YOLO desde los cuadrilateros, con el margen indicado.
    Solo toca las que tienen cuadrilatero: el resto se queda como esta."""
    quads = load_quads()
    if not quads:
        print(f"No hay cuadrilateros en {QUADS}. Marca esquinas con la tecla c.")
        return 1
    n = 0
    for stem, quad in quads.items():
        path, _ = label_path_for(stem)
        if path is None:
            continue
        write_box(path, box_from_quad(quad, pad))
        n += 1
    print(f"{n} cajas rehechas desde el cuadrilatero con margen {pad:.0%}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only-pending", action="store_true",
                        help="solo las imagenes que aun no tienen ninguna caja")
    parser.add_argument("--source", choices=["real_plates", "new_plates"],
                        help="limitar a un dataset")
    parser.add_argument("--redo", action="store_true",
                        help="incluir tambien las ya revisadas")
    parser.add_argument("--review", action="store_true",
                        help="repasar SOLO las ya guardadas, de mas a menos "
                             "sospechosa, para validarlas o rehacerlas")
    parser.add_argument("--all", action="store_true",
                        help="repasar TODAS las que tengan etiqueta, revisadas o "
                             "no, de mas a menos sospechosa")
    parser.add_argument("--newest", type=int, metavar="N",
                        help="repasar solo las N revisadas mas recientemente, "
                             "para comprobar el ultimo lote que se anadio")
    parser.add_argument("--list", action="store_true",
                        help="listar lo que queda por revisar y salir, sin abrir ventana")
    parser.add_argument("--pad", type=float, default=PAD_FRAC,
                        help=f"margen de la caja derivada del cuadrilatero "
                             f"(por defecto {PAD_FRAC})")
    parser.add_argument("--rebuild", action="store_true",
                        help="rehacer las cajas desde los cuadrilateros y salir")
    args = parser.parse_args()

    if not YOLO_DIR.is_dir():
        print(f"ERROR: no encuentro {YOLO_DIR}. Ejecuta antes scripts/06_export_yolo.py")
        return 1

    if args.rebuild:
        return rebuild(args.pad)

    reviewed = load_reviewed()
    quads = load_quads()

    # El ultimo lote: las N con la marca de tiempo mas reciente. Sirve para
    # comprobar de un vistazo que las fotos recien anadidas estan bien nombradas.
    lote = None
    if args.newest:
        lote = {r["stem"] for r in sorted(reviewed.values(),
                                          key=lambda r: r["when"],
                                          reverse=True)[:args.newest]}

    items = []
    for split in ("train", "val"):
        for label in sorted((YOLO_DIR / "labels" / split).glob("*.txt")):
            stem = label.stem
            if lote is not None:
                if stem not in lote:
                    continue
            elif args.all:
                pass                       # todas las que tengan etiqueta
            elif args.review:
                if stem not in reviewed:
                    continue
            elif stem in reviewed and not args.redo:
                continue
            box = read_box(label)
            if args.only_pending and box is not None:
                continue
            path = find_source_image(stem)
            if path is None:
                continue
            source = "real_plates" if stem.startswith("r_") else "new_plates"
            if args.source and source != args.source:
                continue
            width, height = image_size(path)
            items.append({"stem": stem, "split": split, "label": label, "path": path,
                          "source": source, "view": path.parent.name,
                          "plate": path.stem,
                          "ar": box_aspect(box, width, height),
                          "status": reviewed.get(stem, {}).get("status")})

    if lote is not None:
        # Comprobando nombres: el orden alfabetico es mas comodo de seguir.
        items.sort(key=lambda it: it["plate"])
    else:
        # Primero lo mas dudoso: las que no tienen caja, luego las mas achatadas
        # (AR bajo = mucho fondo recogido), luego el resto. En las pre-anotadas es
        # donde mas se nota: una caja del detector con AR 1.2 casi siempre es un
        # faro o una rejilla, no la matricula.
        items.sort(key=lambda it: (it["ar"] is not None, it["ar"] or 0))

    if not items:
        print("No queda nada por revisar.")
        print(f"  {len(reviewed)} imagenes ya revisadas -> {REVIEWED}")
        return 0

    with_box = sum(1 for it in items if it["ar"] is not None)
    ya_vistas = sum(1 for it in items if it["status"])
    print(f"{len(items)} imagenes, ordenadas de mas a menos sospechosa")
    print(f"  {with_box} con caja, {len(items) - with_box} sin ella")
    if ya_vistas:
        print(f"  {ya_vistas} ya revisadas a mano, {len(items) - ya_vistas} "
              f"con la caja PRE-ANOTADA por el detector")
    else:
        print(f"  todas con la caja PRE-ANOTADA por el detector (sin revisar)")
    print(f"  {sum(1 for it in items if it['stem'] in quads)} con cuadrilatero")
    total = sum(len(list((YOLO_DIR / "labels" / s).glob("*.txt")))
                for s in ("train", "val"))
    print(f"En total hay {len(reviewed)} de {total} revisadas.\n")

    if args.list:
        for it in items[:20]:
            mark = f"AR {it['ar']:.2f}" if it["ar"] is not None else "sin caja"
            marca = "quad" if it["stem"] in quads else "rect"
            print(f"  [{mark:>8}] {marca}  {it['split']:<5} {it['source']:<11} "
                  f"{it['view']:<7} {it['plate']}")
        if len(items) > 20:
            print(f"  ... y {len(items) - 20} mas")
        return 0

    if lote is not None:
        print("  COMPRUEBA que la matricula del titulo coincide con la de la foto")
        print("  a correcta | c rehacer la caja | d rehacer rectangulo")
    elif args.review:
        print("  a la caja esta bien | c rehacer por esquinas | d rehacer rectangulo")
    else:
        print("  c esquinas (recomendado) | a aprobar | d rectangulo")
    print("  n sin matricula | s saltar | z deshacer | +/- zoom | q salir\n")

    annotator = Annotator(items, quads, args.pad,
                          review=args.review or lote is not None)
    annotator.run(reviewed)

    counts = {}
    for row in reviewed.values():
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    hechas = sum(len(list((YOLO_DIR / "labels" / s).glob("*.txt")))
                 for s in ("train", "val"))
    print(f"\nrevisadas en total: {len(reviewed)} de {hechas} -> {REVIEWED}")
    for status, n in sorted(counts.items()):
        print(f"  {status:<10} {n}")
    print(f"cuadrilateros: {len(load_quads())} -> {QUADS}")

    # Saltar no guarda nada, y se confunde con aprobar. Avisar aqui, porque si
    # no el repaso parece hecho y en realidad esas imagenes siguen sin revisar.
    pendientes = annotator.skipped - set(reviewed)
    if pendientes:
        print(f"\n  AVISO: {len(pendientes)} saltadas con 's'. NO se han guardado")
        print("  y volveran a salir. Si la caja estaba bien, pulsa 'a', no 's'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
