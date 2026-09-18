"""Deteccion de regiones candidatas a matricula (Sesion 1-2).

Sustituye al modulo `license_plate.LicensePlateDetector` que importa el codigo
base del profesorado y que no venia con el enunciado.

El enfoque es morfologico clasico (Obj2 del challenge): las matriculas son
caracteres oscuros sobre una chapa clara, lo que produce una firma muy
caracteristica de bordes verticales densos y regulares dentro de una region
brillante.
"""

import cv2
import numpy as np

# Ancho de trabajo. Las imagenes del dataset son 4032x2268, pero el coche llena
# el encuadre y la matricula ocupa ~23% del ancho, mucho mas que en la receta
# clasica de este pipeline. Los kernels de abajo actuan a escala de CARACTER, y
# solo funcionan si el caracter mide ~10 px: de ahi 400 y no 1024.
# Medido sobre 12 imagenes del dataset (6 Frontal + 6 Lateral):
#   ancho 320 -> 9/12 placas correctas    ancho 480 -> 10/12
#   ancho 400 -> 10/11 placas correctas   ancho 640 ->  9/12    ancho 1024 -> 6/12
# Por encima de ~640 el recorte parte la matricula por la mitad.
WORK_WIDTH = 400

# Relacion de aspecto admisible (lado largo / lado corto). La matricula espanola
# es 4.73 de frente; la perspectiva lateral la comprime, y un recorte imperfecto
# la alarga, de ahi el rango.
AR_RANGE = (2.5, 6.5)

# Area del candidato como fraccion del area de la imagen. Una matricula ocupa
# ~0.023 en este dataset; el minimo descarta ruido y el maximo, la carroceria.
AREA_FRAC_RANGE = (1e-3, 8e-2)

# Contornos mas grandes que se examinan, y candidatos que se devuelven.
SEARCH_TOP = 15
MAX_CANDIDATES = 10


def normalized_angle(rect):
    """Angulo del lado LARGO del rectangulo respecto a la horizontal, en (-90, 90].

    0 grados = lado largo horizontal = vista frontal.

    `cv2.minAreaRect` no garantiza que el lado que llama "width" sea el mayor,
    asi que su angulo unas veces se mide sobre el lado largo y otras sobre el
    corto. El codigo base corrige eso con `angle - 90 if w < h`
    (data_exploration.py:68-71), pero esa formula asume el rango (0, 90] que
    devuelve OpenCV 4.5-4.x. Otras versiones (entre ellas la 5.0 que usamos)
    devuelven [-90, 0), y entonces la formula da -180 para una placa horizontal.

    Medir el lado largo con atan2 sobre boxPoints no depende de la version.
    """
    box = cv2.boxPoints(rect)
    edges = [box[1] - box[0], box[2] - box[1]]
    long_edge = max(edges, key=np.linalg.norm)
    angle = np.degrees(np.arctan2(long_edge[1], long_edge[0]))
    # La orientacion es modulo 180: plegar al rango (-90, 90].
    if angle > 90:
        angle -= 180
    elif angle <= -90:
        angle += 180
    return float(angle)


def aspect_ratio(rect):
    """Relacion lado largo / lado corto de un minAreaRect."""
    w, h = rect[1]
    lo, hi = min(w, h), max(w, h)
    return hi / lo if lo > 0 else 0.0


# Etiquetas de los pasos intermedios, en orden. Las usa
# scripts/03_explain_pipeline.py para rotular el panel.
STEP_LABELS = [
    ("gray", "1. Gris"),
    ("blackhat", "2. Blackhat = caracteres"),
    ("light", "3. Zonas claras (Otsu)"),
    ("sobel", "4. Sobel-x = bordes verticales"),
    ("closed", "5. Cierre + Otsu = blobs"),
    ("final", "6. Mascara final"),
]


def plate_mask(gray, steps=None):
    """Mascara binaria de las regiones con aspecto de matricula.

    Si `steps` es un dict, se rellena con las imagenes intermedias (claves de
    STEP_LABELS) para poder ilustrar el pipeline sin reimplementarlo. No afecta
    al resultado.
    """
    def record(name, image):
        if steps is not None:
            steps[name] = image.copy()

    record("gray", gray)
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Blackhat: resalta lo oscuro sobre fondo claro -> los caracteres.
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
    record("blackhat", blackhat)

    # Regiones claras de la escena: aisla la chapa blanca de la matricula.
    light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kernel)
    light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    record("light", light)

    # Gradiente vertical: los caracteres generan bordes verticales densos.
    grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    grad = np.abs(grad)
    lo, hi = grad.min(), grad.max()
    grad = np.zeros_like(grad) if hi == lo else (255 * (grad - lo) / (hi - lo))
    grad = grad.astype("uint8")
    record("sobel", grad)

    # Cerrar los huecos entre caracteres para que la matricula sea UN blob.
    grad = cv2.GaussianBlur(grad, (5, 5), 0)
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, rect_kernel)
    thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    record("closed", thresh)

    # Post-proceso: limpiar ruido y quedarse solo con lo que cae en zona clara.
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.bitwise_and(thresh, thresh, mask=light)
    thresh = cv2.dilate(thresh, None, iterations=2)
    thresh = cv2.erode(thresh, None, iterations=1)
    record("final", thresh)

    return thresh


def candidate_verdict(rect, image_area, ar_range=AR_RANGE,
                      area_frac_range=AREA_FRAC_RANGE):
    """Decide si un minAreaRect pasa los filtros: (acepta, motivo del rechazo).

    Es el unico sitio donde vive el criterio de aceptacion; lo usan tanto
    `detectPlates` como el script que ilustra el pipeline.
    """
    frac = (rect[1][0] * rect[1][1]) / image_area
    if not (area_frac_range[0] <= frac <= area_frac_range[1]):
        return False, (f"area {frac:.2%} fuera de "
                       f"[{area_frac_range[0]:.1%}, {area_frac_range[1]:.0%}]")
    ar = aspect_ratio(rect)
    if not (ar_range[0] <= ar <= ar_range[1]):
        return False, f"aspecto {ar:.2f} fuera de [{ar_range[0]}, {ar_range[1]}]"
    return True, ""


def detectPlates(image, work_width=WORK_WIDTH, max_candidates=MAX_CANDIDATES,
                 ar_range=AR_RANGE, area_frac_range=AREA_FRAC_RANGE,
                 search_top=SEARCH_TOP):
    """Devuelve los contornos candidatos a matricula, en coordenadas de `image`.

    La firma es la que espera el codigo base: una lista de contornos aptos para
    `cv2.minAreaRect`, ordenados de mayor a menor plausibilidad (area).
    """
    scale = work_width / image.shape[1]
    small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    mask = plate_mask(gray)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:search_top]

    small_area = small.shape[0] * small.shape[1]

    candidates = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        accepted, _ = candidate_verdict(rect, small_area, ar_range, area_frac_range)
        if not accepted:
            continue
        # Volver a coordenadas de la imagen original: el angulo es invariante a
        # la escala, pero el area no lo seria.
        candidates.append(np.round(contour / scale).astype(np.int32))
        if len(candidates) == max_candidates:
            break

    return candidates
