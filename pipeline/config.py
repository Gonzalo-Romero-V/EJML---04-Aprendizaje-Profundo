"""Constantes y contratos del pipeline.

Centraliza todo lo reproducible del Ejercicio 4 (Aprendizaje Profundo):
semilla, clases del dataset, tamano de imagen, hiperparametros de la CNN propia
y del Transfer Learning (MobileNetV2), y rutas. Cualquier modulo importa de aqui.

Dominio: clasificacion de residuos para reciclaje. Dataset: TrashNet
(garythung/trashnet, Stanford CS229), version redimensionada 512x384. Se usan
5 de las 6 clases (se descarta `trash`, heterogenea y escasa). El par
cardboard/paper se conserva a proposito: es el confundible natural que alimenta
el analisis de errores de la Parte D.
"""
from __future__ import annotations

from pathlib import Path

# ── Reproducibilidad ─────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Dataset (Parte A) ────────────────────────────────────────────────────────
# Fuente: https://github.com/garythung/trashnet  ->  data/dataset-resized.zip
# Descomprime en carpetas por clase; el script de preparacion selecciona estas
# 5 clases y recorta cada una a CAP_POR_CLASE (subsample con semilla fija) para
# dejar el dataset perfectamente balanceado (cumple el maximo 60/40 del enunciado
# con margen) y acelerar el entrenamiento.
CLASES = ["cardboard", "glass", "metal", "paper", "plastic"]
NUM_CLASES = len(CLASES)

# Conteos del TrashNet original: cardboard 403, glass 501, metal 410,
# paper 594, plastic 482. El minimo entre las 5 elegidas es cardboard (403),
# por eso 400 entra holgado en todas y deja 5x400 = 2000 imagenes balanceadas.
CAP_POR_CLASE = 400

# Nombre de la carpeta raiz dentro del zip de TrashNet.
TRASHNET_SUBDIR = "dataset-resized"
TRASHNET_ZIP_URL = (
    "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
)

# ── Carga e imagenes (Parte B.1) ─────────────────────────────────────────────
IMG_SIZE = (128, 128)          # MobileNetV2 tiene pesos ImageNet para 128
IMG_SHAPE = (*IMG_SIZE, 3)
BATCH_SIZE = 32
VAL_SPLIT = 0.20               # split 80/20 train/test exigido por el enunciado

# Data augmentation SOLO en train (nunca en validacion/test). Minimos del
# enunciado: flip horizontal, rotacion, zoom.
AUG_FLIP = "horizontal"
AUG_ROTATION = 0.10            # fraccion de 2*pi (~36 grados)
AUG_ZOOM = 0.10

# ── CNN desde cero (Parte B) ─────────────────────────────────────────────────
# >=3 bloques convolucionales; cada bloque justifica filtros, kernel, BatchNorm
# y Dropout en el informe (la arquitectura concreta vive en cnn.py).
CNN_EPOCHS = 40                # >=30 exigido
CNN_EARLY_STOPPING_PATIENCE = 5
CNN_LR = 1e-3
# Momentum de BatchNormalization. El default de Keras (0.99) tarda demasiados
# pasos en estabilizar las medias moviles con un dataset chico; 0.9 las hace
# converger mas rapido y evita que la validacion colapse al inicio.
BN_MOMENTUM = 0.9
# Regularizacion L2 para la variante mejorada de la CNN (Parte D, mejora 1).
# La CNN base usa 0.0; la mejorada usa este valor.
CNN_L2 = 1e-4

# ── Transfer Learning MobileNetV2 (Parte C) ──────────────────────────────────
# Fase 1 (feature extraction): base congelada, solo la cabeza nueva.
TL_FE_EPOCHS = 10
TL_FE_LR = 1e-3
# Fase 2 (fine-tuning): descongelar las ultimas N capas, learning rate bajo
# para no destruir los pesos preentrenados (ver pregunta tecnica 14).
TL_FT_EPOCHS = 10
TL_FT_LR = 1e-5
TL_FT_UNFREEZE_LAST = 30       # ultimas 30 capas, segun el enunciado

# ── Entorno de computo (local vs Colab) ──────────────────────────────────────
# El codigo es identico en ambos; solo cambian las rutas de datos y el paso de
# instalacion. El notebook detecta Colab y clona el repo + monta el dataset.
try:                                  # pragma: no cover - depende del entorno
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                 # zip descargado + descomprimido
DATASET_DIR = DATA_DIR / "dataset"         # destino final: dataset/<clase>/*.jpg
DOCS_DIR = ROOT / "docs"
FIGURES_DIR = DOCS_DIR / "figuras"
MODELS_DIR = DATA_DIR / "models"           # pesos .keras (no se versionan)
RESULTADOS_JSON = DATA_DIR / "resultados.json"
