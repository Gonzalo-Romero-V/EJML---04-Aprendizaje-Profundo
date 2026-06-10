"""Auditoria y mejora (Parte D).

  10. par_mas_confundido: a partir de la matriz de confusion, el par de clases
      que el modelo confunde mas (mayor celda fuera de la diagonal).
  11. construir_cnn_mejorada + callbacks_mejorados: el modelo con menor F1 (en
      la practica, la CNN propia) se reentrena con DOS mejoras concretas:
        Mejora 1 (regularizacion): L2 en conv y densa (config.CNN_L2).
        Mejora 2 (learning rate schedule): ReduceLROnPlateau ademas de Early
        Stopping, para bajar el lr cuando val_loss se estanca.
      Si mejoraron o no se documenta comparando F1 antes/despues.
  12. recolectar_errores: imagenes mal clasificadas por el mejor modelo, para
      analizar que tienen en comun (ambiguedad).
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from pipeline import cnn, config


def par_mas_confundido(cm: np.ndarray, class_names: list[str]
                       ) -> tuple[str, str, int]:
    """Par (verdad, prediccion) con mas confusiones fuera de la diagonal."""
    fuera = cm.copy()
    np.fill_diagonal(fuera, 0)
    i, j = np.unravel_index(int(fuera.argmax()), fuera.shape)
    return class_names[i], class_names[j], int(fuera[i, j])


def construir_cnn_mejorada(*, l2_reg: float = config.CNN_L2,
                           lr: float = config.CNN_LR) -> tf.keras.Model:
    """CNN con Mejora 1 (regularizacion L2). Misma arquitectura base."""
    return cnn.construir_cnn(l2_reg=l2_reg, lr=lr)


def callbacks_mejorados(patience: int = config.CNN_EARLY_STOPPING_PATIENCE
                        ) -> list[tf.keras.callbacks.Callback]:
    """Early Stopping + Mejora 2 (ReduceLROnPlateau: schedule de learning rate)."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
    ]


def recolectar_predicciones(modelo: tf.keras.Model, ds: tf.data.Dataset
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (imagenes, y_true, y_pred) de todo el dataset en una pasada.

    Conserva las imagenes (uint8, [0,255]) para poder mostrarlas en el analisis
    de errores y en Grad-CAM. El orden coincide con data.cargar_test_crudo (cache).
    """
    imgs, y_true, y_pred = [], [], []
    for xb, yb in ds:
        prob = modelo.predict(xb, verbose=0)
        imgs.append(xb.numpy().astype("uint8"))
        y_true.append(yb.numpy())
        y_pred.append(prob.argmax(axis=1))
    return (np.concatenate(imgs), np.concatenate(y_true),
            np.concatenate(y_pred))


def indices_mal_clasificados(y_true: np.ndarray, y_pred: np.ndarray,
                             *, maximo: int | None = None) -> np.ndarray:
    """Indices donde el modelo se equivoco (para mostrar esas imagenes)."""
    errores = np.flatnonzero(y_true != y_pred)
    return errores if maximo is None else errores[:maximo]
