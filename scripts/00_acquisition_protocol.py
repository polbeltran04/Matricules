"""Sesion 1: identificar el protocolo de adquisicion a partir de los EXIF.

Segundo bullet de la slide 16 del enunciado ("Identify the acquisition protocol
used for the dataset"). Los metadatos EXIF contestan de forma objetiva con que
camara, con que optica y en que condiciones se capturo el dataset, en vez de
deducirlo mirando las fotos.

Uso:
    python scripts/00_acquisition_protocol.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image
from PIL.ExifTags import TAGS

import config
from alpr.dataset import unique_images

# Campos EXIF que describen el protocolo: equipo, optica y condiciones de luz.
FIELDS = ["Model", "FocalLength", "FNumber", "ExposureTime",
          "ISOSpeedRatings", "Flash", "DateTimeOriginal"]


def exif_row(path):
    image = Image.open(path)
    raw = {TAGS.get(k, k): v for k, v in (image._getexif() or {}).items()}
    row = {"file": path.name, "width": image.size[0], "height": image.size[1]}
    row.update({field: raw.get(field) for field in FIELDS})
    return row


def describe_constant(df, column):
    """Un campo constante define el protocolo; uno variable, su margen."""
    values = df[column].dropna()
    if values.empty:
        return f"  {column:<18} (sin dato)"
    counts = Counter(values)
    if len(counts) == 1:
        return f"  {column:<18} {values.iloc[0]}  (constante en las {len(values)} imagenes)"
    top = ", ".join(f"{v} x{n}" for v, n in counts.most_common(3))
    return f"  {column:<18} {len(counts)} valores distintos -> {top}"


def main():
    rows = []
    for view in config.VIEWS:
        directory = config.REAL_PLATES_DIR / view
        if not directory.is_dir():
            continue
        paths, _ = unique_images(directory)
        for path in paths:
            row = exif_row(path)
            row["view"] = view
            rows.append(row)

    if not rows:
        print("no se encontro ninguna imagen")
        return 1

    df = pd.DataFrame(rows)
    config.OUT_DIR.mkdir(exist_ok=True)
    csv_path = config.OUT_DIR / "exif.csv"
    df.to_csv(csv_path, index=False)

    print(f"PROTOCOLO DE ADQUISICION ({len(df)} imagenes unicas)\n")
    print("Equipo y optica:")
    for column in ["Model", "FocalLength", "FNumber", "width", "height"]:
        print(describe_constant(df, column))

    print("\nCondiciones de captura:")
    for column in ["Flash", "ExposureTime"]:
        print(describe_constant(df, column))

    iso = df["ISOSpeedRatings"].dropna()
    if not iso.empty:
        print(f"  {'ISO':<18} {iso.min()} - {iso.max()} (mediana {int(iso.median())})"
              f"   <- a mayor ISO, menos luz disponible")

    dates = df["DateTimeOriginal"].dropna().astype(str)
    if not dates.empty:
        days = sorted({d[:10] for d in dates})
        hours = sorted({d[11:13] for d in dates})
        print(f"  {'Fechas':<18} {dates.min()} -> {dates.max()}")
        print(f"  {'Dias distintos':<18} {len(days)}: {', '.join(days)}")
        print(f"  {'Horas del dia':<18} {', '.join(h + 'h' for h in hours)}")

    print(f"\nReparto por vista: "
          + ", ".join(f"{v}={n}" for v, n in df['view'].value_counts().items()))
    print(f"\ndetalle -> {csv_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
