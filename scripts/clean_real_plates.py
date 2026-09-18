"""Deja `real_plates/` consistente: sin duplicados y con la matricula por nombre.

El dataset del profesorado viene con tres irregularidades que obligan a tratarlo
con cuidado en todos los scripts:

  - 29 ficheros `PXL_*.jpg` son copias byte a byte de los ya nombrados con la
    matricula, lo que sesga cualquier estadistica que no deduplique.
  - 2 imagenes (ya deduplicadas) no llevan matricula en el nombre, asi que no
    tienen ground truth. Sus matriculas se leyeron a mano; estan en MANUAL_PLATES.
  - Hay sufijos sueltos: `3340JMF_.jpg`, `3326HGW - copia.jpg`.

Tras pasar esto, cada fichero es `<MATRICULA>.jpg` y no hay duplicados, con lo
que el ground truth es directo y uniforme.

No hace falta para trabajar: `alpr.dataset.unique_images()` ya deduplica al
vuelo. Esto solo deja el dataset en disco tan limpio como el nuestro.

Uso:
    python scripts/clean_real_plates.py           # dry-run: dice que haria
    python scripts/clean_real_plates.py --apply   # lo hace
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from alpr.dataset import file_md5, plate_from_filename

# Matriculas leidas a mano de las dos imagenes que no la llevan en el nombre.
MANUAL_PLATES = {
    "PXL_20210921_150228605.jpg": "9892JFR",   # Frontal, VW Golf azul
    "PXL_20210921_094938026.jpg": "1556GMZ",   # Lateral, VW Touran negro
}

# Ficheros cuyo nombre no coincide con la matricula que se ve en la foto.
# `3567DCX` era el apano del profesorado para no repetir nombre: hay dos fotos
# laterales del mismo Peugeot 3587 DCX. Aqui se resuelve con sufijo `_2`, que
# `plate_from_filename()` ignora al extraer el ground truth.
RELABEL = {
    "3567DCX.jpg": "3587DCX",   # es el mismo coche que Lateral/3587DCX.jpg
}


def plan_for(directory):
    """Devuelve (renombrados, borrados, sin_resolver) para una carpeta."""
    paths = sorted(directory.glob("*.jpg"))
    # Orden de prioridad para los desempates: primero las que ya llevan bien la
    # matricula (conservan su nombre), luego las reetiquetadas, y al final las
    # que no la llevan. Asi el `_2` recae en la copia, no en el fichero correcto.
    paths.sort(key=lambda p: (plate_from_filename(p) is None, p.name in RELABEL))

    renames, deletes, unresolved = [], [], []
    seen, taken = {}, set()
    for path in paths:
        digest = file_md5(path)
        if digest in seen:
            deletes.append((path, seen[digest]))
            continue
        seen[digest] = path

        plate = (RELABEL.get(path.name) or plate_from_filename(path)
                 or MANUAL_PLATES.get(path.name))
        if plate is None:
            unresolved.append(path)
            continue

        # Dos fotos distintas del mismo coche en la misma carpeta: se desempatan
        # con `_2`, `_3`... El ground truth sigue siendo la matricula, porque
        # `plate_from_filename()` solo mira el principio del nombre.
        stem, n = plate, 2
        while stem in taken:
            stem, n = f"{plate}_{n}", n + 1
        taken.add(stem)

        target = directory / f"{stem}.jpg"
        if target != path:
            renames.append((path, target))
    return renames, deletes, unresolved


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apply", action="store_true",
                        help="ejecutar los cambios (por defecto solo los muestra)")
    args = parser.parse_args()

    if not config.REAL_PLATES_DIR.is_dir():
        print(f"ERROR: no encuentro {config.REAL_PLATES_DIR}")
        return 1

    total_r = total_d = 0
    problems = []

    for view in config.VIEWS:
        directory = config.REAL_PLATES_DIR / view
        if not directory.is_dir():
            continue
        renames, deletes, unresolved = plan_for(directory)

        # Colision: dos imagenes distintas que reclaman el mismo nombre.
        targets = [t for _, t in renames]
        clashes = {t.name for t in targets if targets.count(t) > 1}
        clashes |= {t.name for s, t in renames if t.exists() and t not in [x for x, _ in renames]}
        if clashes:
            problems.append(f"{view}: colision de nombres {sorted(clashes)}")
        problems += [f"{view}: sin matricula conocida -> {p.name}" for p in unresolved]

        print(f"\n{view}: {len(deletes)} duplicados a borrar, {len(renames)} a renombrar")
        for dup, keep in deletes:
            print(f"   borrar   {dup.name}  (copia de {keep.name})")
        for src, dst in renames:
            print(f"   renombrar {src.name}  ->  {dst.name}")

        if args.apply and not clashes:
            for dup, _ in deletes:
                dup.unlink()
            # Renombrar en dos pasos evita pisar un nombre aun ocupado.
            for src, dst in renames:
                src.rename(src.with_suffix(".tmp_rename"))
            for src, dst in renames:
                src.with_suffix(".tmp_rename").rename(dst)
        total_r += len(renames)
        total_d += len(deletes)

    print(f"\n{'APLICADO' if args.apply else 'DRY-RUN'}: "
          f"{total_d} duplicados, {total_r} renombrados")
    for p in problems:
        print(f"  AVISO  {p}")
    if not args.apply:
        print("\nVuelve a lanzarlo con --apply para ejecutarlo.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
