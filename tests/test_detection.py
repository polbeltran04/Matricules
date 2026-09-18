"""Tests de `alpr.detection`.

Ejecutar con:  python tests/test_detection.py
(o con pytest, si esta instalado)

El test del angulo importa: toda la estadistica de viewpoint de la Sesion 1 se
apoya en `normalized_angle`, y la formula del codigo base da -180 grados para
una placa horizontal segun la version de OpenCV.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from alpr.detection import aspect_ratio, detectPlates, normalized_angle


def _rotated_rect_contour(theta_deg, width=520, height=110, center=(600, 400)):
    """Contorno de un rectangulo `width` x `height` rotado `theta_deg` grados."""
    t = np.deg2rad(theta_deg)
    rotation = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    corners = np.array([[-width / 2, -height / 2], [width / 2, -height / 2],
                        [width / 2, height / 2], [-width / 2, height / 2]])
    pts = (corners @ rotation.T) + np.asarray(center)
    return np.round(pts).astype(np.int32).reshape(-1, 1, 2)


def test_normalized_angle_recovers_known_rotation():
    """El angulo medido debe ser el angulo con el que se genero el rectangulo."""
    for theta in [-89, -60, -40, -20, -10, -5, -1, 0, 1, 5, 10, 20, 40, 60, 85, 89]:
        rect = cv2.minAreaRect(_rotated_rect_contour(theta))
        measured = normalized_angle(rect)
        assert abs(measured - theta) < 0.5, f"theta={theta} -> {measured:.2f}"


def test_normalized_angle_is_zero_for_horizontal_plate():
    """Caso critico: una placa frontal debe dar 0, no -180 ni 90."""
    rect = cv2.minAreaRect(_rotated_rect_contour(0))
    assert abs(normalized_angle(rect)) < 0.5


def test_aspect_ratio_is_orientation_independent():
    """La relacion de aspecto no puede depender de como oriente OpenCV el rect."""
    for theta in [0, 30, 60, 90]:
        rect = cv2.minAreaRect(_rotated_rect_contour(theta))
        assert abs(aspect_ratio(rect) - 520 / 110) < 0.1


def _synthetic_plate_image(text="5796DKP", angle=0.0):
    """Imagen sintetica con una chapa clara y caracteres oscuros.

    Las proporciones imitan las del dataset real (la matricula ocupa ~23% del
    ancho de la imagen), porque los kernels morfologicos actuan a escala de
    caracter y un tamano irreal invalida el test.
    """
    h, w = 720, 1280
    image = np.full((h, w, 3), 70, dtype=np.uint8)
    pw, ph = int(w * 0.23), int(w * 0.23 / (520 / 110))
    plate = np.full((ph, pw, 3), 240, dtype=np.uint8)
    cv2.rectangle(plate, (0, 0), (int(pw * 0.09), ph), (150, 60, 20), -1)  # banda UE
    cv2.putText(plate, text, (int(pw * 0.13), int(ph * 0.78)),
                cv2.FONT_HERSHEY_SIMPLEX, ph / 42, (20, 20, 20), max(1, ph // 22))

    rot = cv2.getRotationMatrix2D((pw / 2, ph / 2), -angle, 1.0)
    rot[0, 2] += w / 2 - pw / 2
    rot[1, 2] += h / 2 - ph / 2
    return cv2.warpAffine(plate, rot, (w, h), dst=image,
                          borderMode=cv2.BORDER_TRANSPARENT)


def test_detect_plates_finds_synthetic_plate():
    """La placa sintetica debe detectarse, centrada y con la forma correcta."""
    image = _synthetic_plate_image()
    candidates = detectPlates(image)
    assert candidates, "no se detecto la placa sintetica"

    rect = cv2.minAreaRect(candidates[0])
    (cx, cy), ar = rect[0], aspect_ratio(rect)
    assert abs(cx - 640) < 90 and abs(cy - 360) < 40, f"centro desviado: {rect[0]}"
    assert 2.5 <= ar <= 6.5, f"relacion de aspecto fuera de rango: {ar:.2f}"
    assert abs(normalized_angle(rect)) < 5, "la placa es horizontal"


def test_detect_plates_recovers_rotation():
    """El angulo detectado debe seguir a la rotacion real de la placa."""
    for angle in [-20, -10, 10, 20]:
        candidates = detectPlates(_synthetic_plate_image(angle=angle))
        assert candidates, f"no se detecto la placa rotada {angle} grados"
        measured = normalized_angle(cv2.minAreaRect(candidates[0]))
        assert abs(measured - angle) < 8, f"angle={angle} -> {measured:.1f}"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASS  {test.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {test.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests OK")
    sys.exit(1 if failed else 0)
