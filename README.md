# Challenge #1 — Automatic License Plate Recognition

PSIV2 / Vision & Learning — Universitat Autònoma de Barcelona.

Detección y reconocimiento de matrículas españolas a partir de fotos de vehículos:

```
Image Acquisition → Detection/Localization → Character Segmentation → Recognition
```

## Puesta en marcha

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

En Linux/macOS: `python3.13 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`

## El dataset no está en el repo

`real_plates/` está en `.gitignore` por dos motivos: pesa 151 MB y son fotos de matrículas
reales de terceros, que son un dato personal identificable. Lo distribuye el profesorado.

Descárgalo del Campus Virtual y descomprímelo en la raíz del proyecto con esta estructura:

```
real_plates/
├── Frontal/    # 0216KZP.jpg, 1062FNT_.jpg, ...
└── Lateral/    # 0182GLK.jpg, 0292LKY.jpg, ...
```

**El nombre del fichero es el ground truth de la matrícula.** El sufijo `_` marca los coches que
aparecen en las dos vistas.

> El dataset trae 29 ficheros `PXL_2021*.jpg` que son copias byte a byte de los ya nombrados con
> la matrícula. El código los descarta por MD5: de 98 ficheros quedan **69 imágenes únicas**
> (19 Frontal + 50 Lateral).

## Uso

```powershell
python main.py                 # pipeline completo: tests + las tres etapas (~22 s)
python main.py --list          # qué etapas hay
python main.py explore detect  # solo esas etapas
python main.py --new           # incluir new_plates/ en la exploración
```

`main.py` encadena las etapas, corta si alguna falla y resume tiempos al final. Cada etapa
sigue siendo ejecutable por separado:

| Etapa | Script | Qué hace | Flags propios |
|---|---|---|---|
| `tests` | `tests/test_detection.py` | Tests de `alpr.detection` | — |
| `protocol` | `scripts/00_acquisition_protocol.py` | Protocolo de adquisición (EXIF) | — |
| `explore` | `scripts/01_data_exploration.py` | Propiedades, estadística y figuras | `--new`, `--show` |
| `detect` | `scripts/02_show_detections.py` | Bounding boxes y mosaicos | `-n N` |

Todo lo generado va a `out/`, que tampoco se versiona.

## Estructura

| Ruta | Qué es |
|---|---|
| `main.py` | Punto de entrada: encadena las etapas en orden |
| `config.py` | Rutas relativas al repo y constantes compartidas |
| `alpr/detection.py` | `detectPlates()`, `normalized_angle()`, `plate_mask()` |
| `alpr/dataset.py` | Carga, deduplicado MD5, ground truth, dibujo de cajas |
| `scripts/` | Un script por entregable, numerados por orden de ejecución |
| `tests/` | Tests de `alpr.detection` |
| `new_plates/` | Dataset ampliado (fotos ignoradas, `metadata.csv` sí se versiona) |
| `data_exploration.py` | Código original del profesorado, conservado como referencia |

El código del profesorado importa `license_plate.LicensePlateDetector`, un módulo que no venía con
el enunciado; está reimplementado en `alpr/detection.py`. Por eso `data_exploration.py` no arranca
tal cual: la versión que funciona es `scripts/01_data_exploration.py`.

## Estado — Sesión 1 completa

- Detección: **98.6%** de imágenes con candidato (68/69); ~88% con la matrícula bien localizada.
- El ángulo del `minAreaRect` **separa las dos vistas**: mediana |ángulo| 0.53° en Frontal frente a
  6.48° en Lateral (Mann-Whitney, p = 4.9e-07).
- Color, saturación e iluminación **no** separan las vistas (p > 0.7): las 69 fotos son del mismo
  Pixel 4 XL en el mismo parking. De ahí que el enunciado pida ampliar el dataset.

## Pendiente

Ampliar el dataset (slide 17 del enunciado): fotos en `new_plates/<Frontal|Lateral>/<MATRICULA>.jpg`,
anotadas en `new_plates/metadata.csv`, cubriendo condiciones que **sigan y que no sigan** el
protocolo original — sol, sombra, noche, luz artificial, reflejos, bajo contraste y desenfoque leve.
