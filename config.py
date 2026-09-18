"""Rutas y constantes compartidas por todo el challenge.

Todas las rutas son relativas a la raiz del repo, para que el proyecto funcione
en cualquier maquina (el codigo base del profesor tenia D:\\Teaching\\... hardcodeado).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

REAL_PLATES_DIR = ROOT / "real_plates"   # dataset dado por el profesorado
NEW_PLATES_DIR = ROOT / "new_plates"     # dataset ampliado por nosotros
OUT_DIR = ROOT / "out"                   # figuras y CSV (no versionado)

VIEWS = ["Frontal", "Lateral"]

# Matricula espanola: 520 x 110 mm -> relacion de aspecto ~4.73.
# En vista lateral la perspectiva la comprime, de ahi el rango amplio del filtro.
PLATE_AR = 520 / 110
