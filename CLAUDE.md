# Challenge #1 — Automatic License Plate Recognition (PSIV2 / Vision & Learning, UAB)

Detectar y reconocer matrículas de vehículos a partir de fotos. Pipeline de 4 etapas:

```
Image Acquisition → Detection/Localization → Character Segmentation → Recognition
     (S1)                   (S2)                    (S2-S3)          ├ OCR (Tesseract/EasyOCR)
                                                                     └ HOG/LBP + SVM/NN  (S5-S6)
```

**Objetivos del enunciado:** Obj1 impacto de la adquisición · Obj2 morfología en pre/post-proceso ·
Obj3 YOLO · Obj4 estructuras de imagen (bordes, blobs, segmentación) · Obj5 descriptores (HOG, LBP) ·
Obj6 split simple vs. cross-validation.

**Calendario:** 18-Sep S1 adquisición · 22-Sep S2 detección · 25-Sep S3 GPU/cluster ·
02-Oct S5 reconocimiento ML · 06-Oct S6 reconocimiento DL · 09-Oct S7 bootstrap validation.

## Restricción del enunciado

**Nada de Jupyter Notebooks.** Todo en `.py` (el enunciado especifica Spyder + numpy/pandas/sklearn/PyTorch).

## Ejecutar

```powershell
.\.venv\Scripts\Activate.ps1          # venv con Python 3.13
python scripts/00_acquisition_protocol.py   # protocolo de adquisición (EXIF)
python scripts/01_data_exploration.py       # propiedades + figuras + out/properties.csv
python scripts/02_show_detections.py        # bounding boxes sobre las imágenes
python tests/test_detection.py              # tests (también corre con pytest)
```

`01_data_exploration.py` acepta `--new` (incluir `new_plates/`) y `--show` (abrir las figuras).
Todo lo generado va a `out/`, que no se versiona.

## Estructura

| Ruta | Qué es |
|---|---|
| `config.py` | Rutas relativas al repo y constantes (`VIEWS`, `PLATE_AR`) |
| `alpr/detection.py` | `detectPlates()`, `normalized_angle()`, `plate_mask()` |
| `alpr/dataset.py` | Carga, deduplicado MD5, `plate_from_filename()`, `draw_candidates()` |
| `scripts/` | Un script por entregable, numerados por orden de ejecución |
| `tests/` | Tests de `alpr.detection` |
| `real_plates/` | Dataset del profesorado — **no tocar** |
| `new_plates/` | Dataset ampliado por nosotros + `metadata.csv` |
| `data_exploration.py` | Código original del profesor, **se conserva como referencia** |

Las etapas siguientes irán en `alpr/segmentation.py` (S2-S3) y `alpr/recognition.py` (S5-S6).

## Dataset: lo que hay que saber

**El nombre del fichero ES el ground truth** de la matrícula: `0216KZP.jpg` → `0216KZP`.
Sufijo `_` cuando el mismo coche aparece en Frontal y Lateral (`3340JMF_.jpg`). Eso da etiquetas
gratis para evaluar las etapas 2-4.

**Trampa: 29 duplicados exactos.** Los `PXL_2021*.jpg` son copias byte a byte de los ficheros ya
nombrados con la matrícula. Sin deduplicar, la estadística sale sesgada. Usar siempre
`alpr.dataset.unique_images()`, que deduplica por MD5 y conserva la copia con matrícula en el nombre.

| | Ficheros | **Únicos** |
|---|---|---|
| Frontal | 32 | **19** |
| Lateral | 66 | **50** |
| Total | 98 | **69** |

### Protocolo de adquisición (medido de los EXIF)

Constante en las 69: **Google Pixel 4 XL**, focal 4.38 mm, f/1.73, 4032×2268, **sin flash**.
6 días entre el 21-sep y el 8-oct de 2021, a las 00h/09h/11h/17h/20h. **ISO 39–1809 (mediana 480)**:
ISO alto = parking cubierto con luz artificial, que es donde está tomada la mayoría.

Consecuencia para la memoria: **color, saturación e iluminación NO separan Frontal de Lateral**
(p > 0.7 en los tres) porque todas las fotos comparten cámara y escenario. Lo único que separa las
vistas es el ángulo. Por eso el enunciado pide ampliar el dataset con condiciones distintas.

## Trampas técnicas ya resueltas

**`cv2.minAreaRect` cambia de convención entre versiones de OpenCV** (`[-90,0)` vs `(0,90]`). La
corrección del código base (`angle - 90 if w < h`, [data_exploration.py:68-71](data_exploration.py#L68-L71))
devuelve **-180° para una placa horizontal** con OpenCV 5.0. Usar `alpr.detection.normalized_angle()`,
que mide el lado largo con `atan2` sobre `boxPoints` y no depende de la versión. Hay test que lo cubre.

**`WORK_WIDTH = 400`, no 1024.** Los kernels morfológicos actúan a escala de *carácter*, y en este
dataset el coche llena el encuadre (la matrícula ocupa ~23% del ancho), mucho más que en la receta
clásica. Medido sobre 12 imágenes: ancho 400 → 10/11 placas correctas; ancho 1024 → 6/12, y además
el recorte parte la matrícula por la mitad.

**`np.int0` no existe en numpy ≥ 2** — usar `np.int32` / `np.intp`.

**matplotlib ≥ 3.9**: en `boxplot` el argumento es `tick_labels`, no `labels`.

## Estado actual (Sesión 1 completa)

- Detección: **98.6%** de imágenes con candidato (68/69); ~88% con la matrícula bien localizada
  por inspección visual de `out/detections/mosaico_*.jpg`.
- El ángulo **separa las vistas**: mediana |ángulo| Frontal 0.53° vs Lateral 6.48°,
  Mann-Whitney p = 4.9e-07. Frontal tiene el 79% de imágenes bajo 3°; Lateral solo el 12%.
- Única imagen sin candidato: `Lateral/0907JRF.jpg`.

## Pendiente: ampliar el dataset (slide 17)

Fotos propias o de internet en `new_plates/<Frontal|Lateral>/<MATRICULA>.jpg`, anotadas en
`new_plates/metadata.csv`. El enunciado exige cubrir imágenes que **sigan y que no sigan** el protocolo:

- [ ] Luz solar directa
- [ ] Sombra
- [ ] Noche
- [ ] Luz artificial
- [ ] Reflejos / glare
- [ ] Bajo contraste
- [ ] Ligeramente desenfocadas
- [ ] Viewpoints distintos de los del protocolo original
