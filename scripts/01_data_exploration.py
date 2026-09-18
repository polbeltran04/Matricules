"""Sesion 1: exploracion de las propiedades del dataset.

Reescritura del `data_exploration.py` del profesorado (que se conserva en la
raiz como referencia). Cambios respecto al original:

  - `detectPlates` se implementa en `alpr.detection`; el original importaba un
    modulo que no venia con el enunciado.
  - Rutas desde `config.py`, no `D:\\Teaching\\...`.
  - Descarta los 29 duplicados exactos del dataset antes de calcular nada.
  - Convierte a HSV una sola vez por imagen, no tres.
  - Vuelca `out/properties.csv` para no recalcular la deteccion al retocar un
    grafico.
  - Contrasta las dos vistas con Mann-Whitney en vez de solo dibujarlas.

Uso:
    python scripts/01_data_exploration.py            # solo real_plates
    python scripts/01_data_exploration.py --new      # incluye new_plates
    python scripts/01_data_exploration.py --show     # ademas abre las figuras
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats

import config
from alpr.dataset import plate_from_filename, unique_images
from alpr.detection import detectPlates, normalized_angle

COLORS = ["b", "r"]


def image_properties(image):
    """Color, saturacion e iluminacion medias (canales H, S, V)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, sat, val = (float(hsv[:, :, c].mean()) for c in range(3))
    return hue, sat, val


def collect(roots):
    """Recorre el dataset y devuelve una fila de propiedades por imagen."""
    rows = []
    for root in roots:
        for view in config.VIEWS:
            directory = root / view
            if not directory.is_dir():
                continue
            paths, n_dups = unique_images(directory)
            print(f"{root.name}/{view}: {len(paths)} imagenes unicas "
                  f"({n_dups} duplicados descartados)")

            for path in paths:
                image = cv2.imread(str(path))
                if image is None:
                    print(f"  aviso: no se pudo leer {path.name}")
                    continue

                hue, sat, val = image_properties(image)
                candidates = detectPlates(image)
                # El primer candidato es el mayor: el mas plausible como placa.
                rect = cv2.minAreaRect(candidates[0]) if candidates else None

                rows.append({
                    "file": path.name,
                    "source": root.name,
                    "view": view,
                    "plate": plate_from_filename(path),
                    "hue": hue,
                    "saturation": sat,
                    "value": val,
                    "n_candidates": len(candidates),
                    "plate_area": float(np.prod(rect[1])) if rect else np.nan,
                    "plate_angle": normalized_angle(rect) if rect else np.nan,
                })
    return pd.DataFrame(rows)


def compare_distributions(df, column, title, groups, out_dir, show):
    """Histograma + boxplot de `column` comparando los valores de `groups`.

    Sustituye a los tres bloques de plotting copiados del script original.
    """
    series = [df.loc[df[groups] == g, column].dropna() for g in sorted(df[groups].unique())]
    labels = sorted(df[groups].unique())

    fig, (ax_hist, ax_box) = plt.subplots(1, 2, figsize=(11, 4))
    for i, values in enumerate(series):
        ax_hist.hist(values, bins=20, edgecolor="k",
                     color=COLORS[i % len(COLORS)], alpha=1 - 0.4 * i, label=labels[i])
    ax_hist.set_title(f"{title} - histograma")
    ax_hist.legend()

    # matplotlib >= 3.9 renombro `labels` a `tick_labels` en boxplot.
    ax_box.boxplot(series, tick_labels=labels)
    ax_box.set_title(f"{title} - boxplot")

    fig.tight_layout()
    fig.savefig(out_dir / f"{column}_{groups}.png", dpi=120)
    if not show:
        plt.close(fig)


def report_separation(df, column, groups="view"):
    """Contrasta si `column` separa los grupos (Mann-Whitney, dos colas)."""
    labels = sorted(df[groups].unique())
    if len(labels) != 2:
        return
    a = df.loc[df[groups] == labels[0], column].dropna()
    b = df.loc[df[groups] == labels[1], column].dropna()
    if len(a) < 3 or len(b) < 3:
        return
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    verdict = "SEPARA" if p < 0.05 else "no separa"
    print(f"  {column:>12}: mediana {labels[0]}={a.median():7.2f}  "
          f"{labels[1]}={b.median():7.2f}  p={p:.2e}  -> {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", action="store_true",
                        help="incluir tambien new_plates/ (dataset ampliado)")
    parser.add_argument("--show", action="store_true",
                        help="mostrar las figuras ademas de guardarlas")
    args = parser.parse_args()

    roots = [config.REAL_PLATES_DIR]
    if args.new:
        if config.NEW_PLATES_DIR.is_dir():
            roots.append(config.NEW_PLATES_DIR)
        else:
            print(f"aviso: {config.NEW_PLATES_DIR} no existe, se ignora --new")

    config.OUT_DIR.mkdir(exist_ok=True)
    df = collect(roots)
    if df.empty:
        print("no se encontro ninguna imagen")
        return 1

    csv_path = config.OUT_DIR / "properties.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n{len(df)} imagenes -> {csv_path}")

    detected = df["n_candidates"].gt(0).mean()
    print(f"tasa de deteccion: {detected:.1%} "
          f"({df['n_candidates'].gt(0).sum()}/{len(df)} imagenes con candidato)")

    properties = [("hue", "Color (H)"),
                  ("saturation", "Saturacion (S)"),
                  ("value", "Iluminacion (V)"),
                  ("plate_angle", "Viewpoint (angulo)"),
                  ("plate_area", "Distancia focal (area)")]

    print("\nSeparacion entre vistas (Mann-Whitney):")
    df["abs_angle"] = df["plate_angle"].abs()
    for column in ["abs_angle"] + [c for c, _ in properties]:
        report_separation(df, column)

    for column, title in properties:
        compare_distributions(df, column, title, "view", config.OUT_DIR, args.show)

    # Con el dataset ampliado, la comparacion que pide el Obj1: nuestras fotos
    # siguen o no el protocolo de adquisicion del profesorado?
    if df["source"].nunique() > 1:
        print("\nSeparacion entre datasets (protocolo vs. dataset ampliado):")
        for column in ["abs_angle"] + [c for c, _ in properties]:
            report_separation(df, column, groups="source")
        for column, title in properties:
            compare_distributions(df, column, title, "source", config.OUT_DIR, args.show)

    print(f"figuras -> {config.OUT_DIR}")

    if args.show:
        plt.show()
    return 0


if __name__ == "__main__":
    sys.exit(main())
