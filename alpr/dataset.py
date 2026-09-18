"""Carga del dataset, deduplicado y dibujo de candidatos.

El nombre del fichero ES el ground truth de la matricula (`0216KZP.jpg`), con
sufijo `_` cuando el mismo coche aparece en Frontal y en Lateral.

Ojo con los duplicados: `real_plates` trae 29 ficheros `PXL_2021*.jpg` que son
copias byte a byte de los ya nombrados con la matricula. Si no se descartan, la
estadistica descriptiva sale sesgada (Frontal pasa de 19 imagenes reales a 32).
"""

import hashlib
import re

import cv2
import numpy as np

# Matricula espanola moderna: 4 digitos + 3 consonantes.
PLATE_RE = re.compile(r"^(\d{4}[A-Z]{3})")


def file_md5(path):
    """Hash del contenido del fichero, para detectar duplicados exactos."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def plate_from_filename(path):
    """Matricula codificada en el nombre, o None si el nombre no la lleva.

    Tolera los sufijos que usa el dataset: `3340JMF_.jpg`, `3326HGW - copia.jpg`.
    Los `PXL_2021*.jpg` no llevan matricula y devuelven None.
    """
    match = PLATE_RE.match(path.stem.upper().strip())
    return match.group(1) if match else None


def unique_images(directory):
    """Imagenes de `directory` sin duplicados exactos, y cuantos se descartaron.

    Ante un duplicado se conserva la copia cuyo nombre lleva la matricula, que
    es la que tiene ground truth.
    """
    paths = sorted(directory.glob("*.jpg"))
    # Las que tienen matricula en el nombre van primero, asi ganan el desempate.
    paths.sort(key=lambda p: plate_from_filename(p) is None)

    seen = {}
    for path in paths:
        seen.setdefault(file_md5(path), path)

    kept = sorted(seen.values())
    return kept, len(paths) - len(kept)


def draw_candidates(image, contours, angles=None, color=(0, 255, 0)):
    """Dibuja el minAreaRect de cada candidato, anotado con su angulo.

    Es el 3er bullet de la slide 16 del enunciado, que el codigo base no hacia.
    """
    canvas = image.copy()
    thickness = max(2, image.shape[1] // 500)
    for i, contour in enumerate(contours):
        rect = cv2.minAreaRect(contour)
        box = np.round(cv2.boxPoints(rect)).astype(np.int32)
        cv2.drawContours(canvas, [box], -1, color, thickness)
        if angles is not None:
            label = f"{angles[i]:+.1f} deg"
            cv2.putText(canvas, label, tuple(box[box[:, 1].argmin()]),
                        cv2.FONT_HERSHEY_SIMPLEX, thickness / 3, color,
                        thickness, cv2.LINE_AA)
    return canvas
