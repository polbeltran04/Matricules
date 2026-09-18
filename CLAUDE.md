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
.\.venv\Scripts\Activate.ps1   # venv con Python 3.13
python main.py                 # pipeline completo: tests + las tres etapas (~22 s)
python main.py --list          # qué etapas hay
python main.py explore detect  # solo esas etapas
python main.py --new           # incluir new_plates/ en la exploración
```

`main.py` encadena las etapas, corta si alguna falla y resume tiempos al final. Cada etapa
**sigue siendo ejecutable por separado** (el enunciado evalúa los entregables uno a uno):

| Etapa | Script | Flags propios |
|---|---|---|
| `tests` | `tests/test_detection.py` | — |
| `dataset` | `tests/test_dataset.py` | — |
| `protocol` | `scripts/00_acquisition_protocol.py` | — |
| `explore` | `scripts/01_data_exploration.py` | `--new`, `--show` |
| `detect` | `scripts/02_show_detections.py` | `-n N` |
| `explain` | `scripts/03_explain_pipeline.py` | `<imagen>`, `--zoom` |

`03_explain_pipeline.py` ilustra el pipeline sobre una imagen: panel de las 6 etapas de
`plate_mask()` y tabla de decisión contorno a contorno. **Reutiliza `plate_mask(steps=...)` y
`candidate_verdict()`**, así que nunca se desincroniza del detector real — si cambias el pipeline,
la explicación cambia sola. `candidate_verdict()` es el único sitio donde vive el criterio de
aceptación.

Todo lo generado va a `out/`, que no se versiona.

## Estructura

| Ruta | Qué es |
|---|---|
| `main.py` | Punto de entrada: encadena las etapas (lista `STAGES`, ampliar ahí al añadir sesiones) |
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

- **Cobertura: 99.0%** (200/202 imágenes con ≥1 candidato), mediana de 3 candidatos por imagen.
- El ángulo **separa las vistas** y la iluminación **separa los datasets** (ver arriba).

### Cobertura ≠ acierto. Son 99.0% y 67.0%

`cobertura` mide en cuántas imágenes `detectPlates` devolvió *algo*, no si era la matrícula: un
faro cuenta igual. Con mediana de 3 candidatos por imagen, la mayoría no lo es.

**El acierto real está medido**: `scripts/05_detection_accuracy.py` (etapa `accuracy`), sobre los
veredictos de `validation/detection_verdicts.csv` — las 200 imágenes con candidato, revisadas una
a una comparando el recorte del candidato #1 con la matrícula del nombre.

| Grupo | Acierto | IC 95% |
|---|---|---|
| **TOTAL** | **67.0%** (134/200) | 60.2 – 73.1% |
| `real_plates` | **86.8%** (59/68) | 76.7 – 92.9% |
| `new_plates` | **56.8%** (75/132) | 48.3 – 65.0% |
| `new_plates` Lateral | **43.3%** (13/30) | 27.4 – 60.8% |

**Ese contraste es el resultado importante de la Sesión 1.** El detector morfológico funciona
razonablemente en el parking controlado del profesorado (87%) y se desploma en fotos de exterior
con sol (57%), hasta el 43% en las laterales. No generaliza — y ahí está la justificación
cuantitativa para pasar a YOLO en la Sesión 2.

El ~88% citado antes de medir esto era una impresión visual sobre `real_plates` solo; casualmente
coincide con el 86.8% real de ese subconjunto, pero no valía como evidencia.

**Limitación**: acierto = el recorte contiene la matrícula entera y legible. **No es IoU**, no mide
lo ajustada que está la caja.

## Anotación de cajas (preparación de la Sesión 2)

`scripts/06_export_yolo.py` genera `yolo_dataset/` en formato estándar de ultralytics, usando los
veredictos como **pre-anotación**: escribe la caja del detector en las 134 que acierta y deja
vacías las 66 que falla, listadas en `PENDIENTES.txt`. Split estratificado por dataset y vista,
semilla fija (160 train / 40 val).

```powershell
python scripts/06_export_yolo.py                # etiquetas
python scripts/06_export_yolo.py --copy-images  # y copia las fotos
```

**Nunca pisa una etiqueta existente** (hace falta `--force`): las correcciones manuales son trabajo
irrecuperable. Las etiquetas **sí se versionan**, las imágenes no.

Dos avisos:

- Una pre-anotación **no es una anotación**. La caja del morfológico contiene la matrícula pero
  suele venir holgada o cortada, y una caja mal ajustada entrena mal. Hay que repasarlas **todas**
  en un editor (labelImg, CVAT, Roboflow, Label Studio).
- Con 202 imágenes no se entrena YOLO de cero: toca *fine-tuning* de un modelo preentrenado.

Cuando las cajas estén revisadas, se podrá medir IoU y comparar morfológico vs. YOLO sobre el mismo
conjunto de validación — que es el Obj3 y, con el split, el Obj6.

### Dos fotos del mismo coche: sufijo `_2`, no inventar matrícula

`plate_from_filename()` usa un regex **anclado al inicio**, así que ignora cualquier sufijo:
`3587DCX_2.jpg` → `3587DCX`. Eso permite tener varias fotos de un mismo coche en la misma carpeta
sin perder el ground truth, y `clean_real_plates.py` asigna los sufijos solo al detectar colisión.

El dataset original no lo hacía y por eso traía dos etiquetas mal, ambas ya corregidas:

| Fichero original | Matrícula real | Ahora |
|---|---|---|
| `3040JMB.jpg` | 3044 JMB | `3044JMB.jpg` |
| `3567DCX.jpg` | 3587 DCX (2ª foto del mismo Peugeot) | `3587DCX_2.jpg` |

**Nunca renombrar inventando una matrícula para esquivar una colisión**: rompe el ground truth de
forma silenciosa, y es justo lo que pasó con `3567DCX`.

## Dataset ampliado (slide 17)

**138 fotos propias** en `new_plates/`, en dos lotes, anotadas en `new_plates/metadata.csv`.
Todas con EXIF intacto (**importante: transferir sin WhatsApp**, que borra el EXIF y baja a
2000×1500; enviar como *Documento* o en ZIP).

| | Lote 1 | Lote 2 |
|---|---|---|
| Cámara | OPPO A94 5G | Xiaomi 2209116AG |
| Fotos | 30 | 108 |
| ISO | 100–2163 | 50 (constante) |

Reparto: **Frontal 103 · Lateral 29 · OtrosFormatos 6**. Total con `real_plates`: **201 imágenes**.

Matrículas leídas a mano y validadas contra el formato español (4 dígitos + 3 consonantes,
sin vocales ni Ñ/Q). **Ese validador detectó dos errores de lectura reales**: `5241OGG`→`5241DGG`
y `6158CCQ`→`6158CCG`, porque la O y la Q no existen en matrículas españolas.

`new_plates/OtrosFormatos/` guarda 5 matrículas que **no** siguen el formato español moderno:
2 francesas, 1 andorrana, 1 británica y 1 española antigua (`B 2048 UJ`, placa blanca sin banda
europea). No está en `config.VIEWS`, así que queda fuera de la estadística y del ground truth,
pero documentada: es un caso real que un ALPR desplegado en España se encuentra.

### Lección: la excepción manual eludió el control

`8222BLZ` se leyó como `B222BLZ` y se clasificó a mano en `OtrosFormatos` por parecer una
matrícula antigua. Al tratarla como excepción, **se saltó el validador de formato** — que es
justo lo que la habría cazado. Otras dos (`4326FGC` y `9935FZK`) se corrigieron porque el usuario
las revisó, no porque ningún control las detectara.

De ahí `tests/test_dataset.py` (etapa `dataset` de `main.py`), que comprueba la integridad del
ground truth de forma automática: formato de todos los nombres, que nada en `OtrosFormatos` sea
en realidad una matrícula española válida, y que `metadata.csv` y el disco coincidan.

**Tasa de error conocida del etiquetado: 3 de 108 (≈3%) en el segundo lote.** Es una cota inferior:
solo son los errores que alguien llegó a revisar. Conviene tenerlo presente al interpretar la
precisión del OCR en las Sesiones 5-6 — parte del error medido será de las etiquetas.

También: tras deduplicar, **2 de las 69 imágenes del profesorado no llevan matrícula en el nombre**
(`PXL_*` sin copia nombrada), así que no tienen ground truth. El test lo deja fijado.

Con `--new`, el script 01 añade la comparación **protocolo vs. dataset ampliado** (Obj1):

| Propiedad | real_plates | new_plates | p | |
|---|---|---|---|---|
| Iluminación (V) | 108.2 | **143.1** | **1.4e-26** | SEPARA |
| Saturación (S) | 41.8 | **49.3** | 2.0e-06 | SEPARA |
| Área de placa | 219432 | 173956 | 2.9e-02 | SEPARA |
| Ángulo | 4.2° | 2.2° | 0.14 | no separa |

Las nuestras son **mucho más brillantes y saturadas, y tomadas desde más lejos**: exterior soleado
frente a parking cubierto. Ese contraste medido es el argumento del Obj1.

Cobertura de condiciones — **falta lo nocturno y el desenfoque**:

- [x] Luz solar directa · sombra · luz artificial (garaje)
- [x] Reflejos / glare, bajo contraste (carrocerías oscuras)
- [x] Viewpoints y cámaras distintas (3 móviles en total)
- [x] Matrículas traseras — el dataset original solo tiene delanteras
- [ ] **Noche**
- [ ] **Desenfocadas**

**Aviso: el reparto Frontal/Lateral está desbalanceado** (103 vs 29). Al evaluar por vista, usar
métricas que no se dejen arrastrar por el desequilibrio.
