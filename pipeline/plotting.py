"""Generacion de figuras — Vista.

Todas las funciones reciben datos/modelos ya calculados, renderizan con
matplotlib (backend Agg, sin ventana) y guardan en docs/figuras/, devolviendo
el Path. Cubren los graficos exigidos: curvas de la CNN, curvas de TL Fases 1 y
2, matrices de confusion, grilla Grad-CAM, tabla comparativa e imagenes mal
clasificadas.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pipeline import config, gradcam  # noqa: E402


def _guardar(fig, nombre: str) -> Path:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return ruta


def curvas_entrenamiento(history: dict, nombre: str, titulo: str, *,
                         linea_finetuning: int | None = None) -> Path:
    """Curvas de accuracy y loss (train vs val) en dos paneles.

    linea_finetuning: epoca donde arranca la Fase 2 (dibuja una vertical para
    separar Feature Extraction de Fine-Tuning en las curvas de TL).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    epocas = range(1, len(history["accuracy"]) + 1)
    ax1.plot(epocas, history["accuracy"], label="train")
    ax1.plot(epocas, history["val_accuracy"], label="val")
    ax1.set_title("Accuracy"); ax1.set_xlabel("epoca"); ax1.legend()
    ax2.plot(epocas, history["loss"], label="train")
    ax2.plot(epocas, history["val_loss"], label="val")
    ax2.set_title("Loss"); ax2.set_xlabel("epoca"); ax2.legend()
    if linea_finetuning is not None:
        for ax in (ax1, ax2):
            ax.axvline(linea_finetuning + 0.5, color="gray", ls="--", lw=1)
    fig.suptitle(titulo)
    return _guardar(fig, nombre)


def matriz_confusion_fig(cm: np.ndarray, class_names: list[str],
                         nombre: str, titulo: str) -> Path:
    """Matriz de confusion como heatmap anotado."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("prediccion"); ax.set_ylabel("verdad"); ax.set_title(titulo)
    umbral = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > umbral else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    return _guardar(fig, nombre)


def grid_gradcam(modelo, imagenes: np.ndarray, class_names: list[str],
                 y_true: np.ndarray, y_pred: np.ndarray, nombre: str, *,
                 capa_conv: str = None) -> Path:
    """Por cada imagen: original | Grad-CAM superpuesto, con etiqueta y prediccion."""
    n = len(imagenes)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    if n == 1:
        axes = axes[None, :]
    kwargs = {"capa_conv": capa_conv} if capa_conv else {}
    for k in range(n):
        img = imagenes[k]
        mapa, _ = gradcam.heatmap(modelo, img.astype("float32"), **kwargs)
        sup = gradcam.superponer(img, mapa)
        axes[k, 0].imshow(img.astype("uint8")); axes[k, 0].axis("off")
        axes[k, 0].set_title(f"real: {class_names[y_true[k]]}")
        axes[k, 1].imshow(sup); axes[k, 1].axis("off")
        axes[k, 1].set_title(f"Grad-CAM -> {class_names[y_pred[k]]}")
    fig.suptitle("Grad-CAM: zonas que activan la red")
    fig.tight_layout()
    return _guardar(fig, nombre)


def tabla_comparativa_fig(df: pd.DataFrame, nombre: str) -> Path:
    """Renderiza la tabla comparativa de los 3 modelos como figura."""
    fig, ax = plt.subplots(figsize=(9, 1.2 + 0.5 * len(df)))
    ax.axis("off")
    tabla = ax.table(cellText=df.round(4).values, colLabels=df.columns,
                     loc="center", cellLoc="center")
    tabla.auto_set_font_size(False); tabla.set_fontsize(10); tabla.scale(1, 1.6)
    ax.set_title("Comparativa: CNN propia vs MobileNetV2 (Fase 1 y 2)", pad=12)
    return _guardar(fig, nombre)


def grid_imagenes(imagenes: np.ndarray, titulos: list[str], nombre: str,
                  titulo: str, *, columnas: int = 4) -> Path:
    """Grilla de imagenes (para las mal clasificadas del mejor modelo)."""
    n = len(imagenes)
    filas = (n + columnas - 1) // columnas
    fig, axes = plt.subplots(filas, columnas, figsize=(3 * columnas, 3 * filas))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        axes[k].axis("off")
        if k < n:
            axes[k].imshow(imagenes[k].astype("uint8"))
            axes[k].set_title(titulos[k], fontsize=9)
    fig.suptitle(titulo)
    fig.tight_layout()
    return _guardar(fig, nombre)
