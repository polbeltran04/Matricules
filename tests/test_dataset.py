"""Comprobaciones de integridad del ground truth.

El nombre de cada fichero es la matricula, asi que un nombre mal escrito es un
error de etiqueta que contaminara la evaluacion del OCR en las Sesiones 5-6.

Estos tests existen por un fallo real: una matricula (8222BLZ) se leyo como
`B222BLZ` y se clasifico a mano en OtrosFormatos, con lo que **se salto el
validador de formato**. La excepcion manual eludio el control automatico. Ahora
el validador se aplica a todo lo que dice ser espanol, venga de donde venga.

Se saltan solos si el dataset no esta descargado (no se versiona).

Ejecutar con:  python tests/test_dataset.py
"""

import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from alpr.dataset import plate_from_filename, unique_images

# Matricula espanola moderna: 4 digitos + 3 consonantes (sin vocales, sin N ni Q).
PLATE_RE = re.compile(r"^\d{4}[BCDFGHJKLMNPRSTVWXYZ]{3}$")

# Carpeta para matriculas que no siguen ese formato (extranjeras, antiguas).
OTHER = "OtrosFormatos"

# Tras pasar scripts/clean_real_plates.py, toda imagen del profesorado lleva su
# matricula en el nombre. Antes eran 2 las que no (`PXL_*` sin copia nombrada).
REAL_WITHOUT_GROUND_TRUTH = 0


def _photos(directory):
    return sorted(directory.glob("*.jpg")) if directory.is_dir() else []


def _plate_of(path):
    """Matricula del nombre, sin el sufijo `_2` que desempata repeticiones."""
    return path.stem.split("_")[0]


def test_new_plates_filenames_are_valid_spanish_plates():
    """Todo fichero nuestro en Frontal/Lateral debe ser una matricula valida.

    Este es el test que habria cazado la lectura erronea `B222BLZ`, que en
    realidad era `8222BLZ`, si se hubiera aplicado en su momento.
    """
    offenders = [f"{root.name}/{view}/{path.name}"
                 for root in (config.NEW_PLATES_DIR, config.REAL_PLATES_DIR)
                 for view in config.VIEWS
                 for path in _photos(root / view)
                 if not PLATE_RE.match(_plate_of(path))]
    assert not offenders, f"nombres que no son matriculas validas: {offenders}"


def test_real_plates_ground_truth_coverage():
    """Cuantas imagenes del profesorado se quedan sin ground truth.

    No es un fallo nuestro, es una propiedad del dataset que conviene tener
    fijada: si el numero cambia, es que el dataset ya no es el mismo.
    """
    if not config.REAL_PLATES_DIR.is_dir():
        return
    without = [p.name for view in config.VIEWS
               for p in unique_images(config.REAL_PLATES_DIR / view)[0]
               if plate_from_filename(p) is None]
    assert len(without) == REAL_WITHOUT_GROUND_TRUTH, (
        f"esperadas {REAL_WITHOUT_GROUND_TRUTH} sin matricula, hay {len(without)}: {without}")


def test_other_formats_are_not_valid_spanish_plates():
    """Nada en OtrosFormatos debe cumplir el formato espanol moderno.

    Si lo cumple, es que se clasifico ahi por error de lectura y deberia estar
    en Frontal o Lateral, contando como ground truth.
    """
    misplaced = [p.name for p in _photos(config.NEW_PLATES_DIR / OTHER)
                 if PLATE_RE.match(_plate_of(p))]
    assert not misplaced, f"son espanolas validas, no van en {OTHER}: {misplaced}"


def test_metadata_matches_files_on_disk():
    """Cada foto de new_plates tiene fila en metadata.csv, y al reves."""
    meta = config.NEW_PLATES_DIR / "metadata.csv"
    if not meta.exists():
        return
    with open(meta, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("filename")]
    if not rows:
        return

    on_disk = {(v, p.name) for v in list(config.VIEWS) + [OTHER]
               for p in _photos(config.NEW_PLATES_DIR / v)}
    if not on_disk:
        return

    in_csv = {(r["view"], r["filename"]) for r in rows}
    assert not (on_disk - in_csv), f"fotos sin fila en metadata.csv: {sorted(on_disk - in_csv)}"
    assert not (in_csv - on_disk), f"filas de metadata.csv sin foto: {sorted(in_csv - on_disk)}"


def test_metadata_plate_matches_filename():
    """La columna `plate` debe coincidir con el nombre del fichero."""
    meta = config.NEW_PLATES_DIR / "metadata.csv"
    if not meta.exists():
        return
    with open(meta, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("filename")]

    mismatched = [f"{r['filename']} dice plate={r['plate']}" for r in rows
                  if r["plate"].replace(" ", "") != Path(r["filename"]).stem.split("_")[0]]
    assert not mismatched, f"plate no coincide con el nombre: {mismatched}"


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
