"""Descarga y preparacion del dataset (Parte A) — entregable de codigo.

Fuente: TrashNet (garythung/trashnet, Stanford CS229), version redimensionada
512x384 incluida en el propio repo como data/dataset-resized.zip. No requiere
autenticacion (a diferencia de Kaggle) y la cita es inequivoca.

Pasos:
  1. Descarga el zip a data/raw/ (omite si ya existe).
  2. Lo descomprime -> data/raw/dataset-resized/<clase>/*.jpg
  3. Selecciona las 5 clases de config.CLASES, recorta cada una a
     config.CAP_POR_CLASE (subsample con semilla fija) y copia a
     data/dataset/<clase>/  -> dataset balanceado, listo para Keras.

Solo stdlib: corre sin el venv / sin TensorFlow.

    Uso:  python scripts/preparar_dataset.py
"""
from __future__ import annotations

import random
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

# La consola de Windows usa cp1252 y no imprime acentos/simbolos por defecto.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

# Permite `python scripts/preparar_dataset.py` desde la raiz del ejercicio.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import config  # noqa: E402

_IMG_EXT = {".jpg", ".jpeg", ".png"}


def descargar_zip(destino: Path) -> None:
    """Descarga el zip de TrashNet si aun no esta en disco."""
    if destino.exists():
        print(f"  · zip ya presente ({destino.stat().st_size/1e6:.1f} MB), se omite descarga")
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"  · descargando {config.TRASHNET_ZIP_URL}")
    req = urllib.request.Request(
        config.TRASHNET_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(destino, "wb") as f:
        shutil.copyfileobj(resp, f)
    print(f"    -> {destino.relative_to(config.ROOT)} ({destino.stat().st_size/1e6:.1f} MB)")


def descomprimir(zip_path: Path, destino: Path) -> Path:
    """Descomprime el zip y devuelve la carpeta raiz con las clases."""
    raiz = destino / config.TRASHNET_SUBDIR
    if raiz.exists():
        print(f"  · ya descomprimido en {raiz.relative_to(config.ROOT)}, se omite")
        return raiz
    print(f"  · descomprimiendo en {destino.relative_to(config.ROOT)}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(destino)
    if not raiz.exists():
        raise FileNotFoundError(
            f"No se encontro {raiz} tras descomprimir; revisar estructura del zip.")
    return raiz


def construir_dataset(origen: Path) -> dict[str, int]:
    """Selecciona, balancea y copia las clases a data/dataset/<clase>/."""
    rng = random.Random(config.RANDOM_STATE)
    if config.DATASET_DIR.exists():
        shutil.rmtree(config.DATASET_DIR)        # reconstruccion idempotente
    conteos: dict[str, int] = {}
    for clase in config.CLASES:
        src = origen / clase
        if not src.exists():
            raise FileNotFoundError(f"Clase '{clase}' no encontrada en {src}")
        imagenes = sorted(p for p in src.iterdir() if p.suffix.lower() in _IMG_EXT)
        if len(imagenes) < config.CAP_POR_CLASE:
            raise ValueError(
                f"Clase '{clase}' tiene {len(imagenes)} imagenes (< CAP "
                f"{config.CAP_POR_CLASE}); ajustar config.CAP_POR_CLASE.")
        rng.shuffle(imagenes)
        seleccion = imagenes[:config.CAP_POR_CLASE]
        destino = config.DATASET_DIR / clase
        destino.mkdir(parents=True, exist_ok=True)
        for img in seleccion:
            shutil.copy2(img, destino / img.name)
        conteos[clase] = len(seleccion)
    return conteos


def main() -> None:
    print("[Parte A] Preparacion del dataset de residuos (TrashNet)")
    zip_path = config.RAW_DIR / "dataset-resized.zip"
    descargar_zip(zip_path)
    raiz = descomprimir(zip_path, config.RAW_DIR)
    conteos = construir_dataset(raiz)

    total = sum(conteos.values())
    print("\n  Dataset final (data/dataset/<clase>/):")
    for clase, n in conteos.items():
        print(f"      {clase:<10} {n} imagenes")
    print(f"      {'TOTAL':<10} {total} imagenes en {len(conteos)} clases")
    minimo, maximo = min(conteos.values()), max(conteos.values())
    print(f"  Balance: min={minimo}, max={maximo}  "
          f"(ratio mayoritaria {maximo/total*100:.0f}% — limite 60/40)")
    print(f"\n✓ Listo. Cargar con image_dataset_from_directory sobre "
          f"{config.DATASET_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
