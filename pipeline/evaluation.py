"""Evaluacion de modelos (Partes B.4 y C.8-9).

Funciones agnosticas al modelo: sirven igual para la CNN propia y para
MobileNetV2 (Fases 1 y 2). Producen estructuras de datos (no figuras): las
metricas por clase, la matriz de confusion (array) y la tabla comparativa
unificada. El renderizado (matriz como heatmap, etc.) vive en plotting.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score)


def predecir(modelo: tf.keras.Model, ds: tf.data.Dataset
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Devuelve (y_true, y_pred, y_prob) recorriendo el dataset una vez.

    Usar sobre el test SIN shuffle (data.cargar_test_crudo) para que el orden
    sea estable y reutilizable por Grad-CAM y el analisis de errores.
    """
    y_true, y_prob = [], []
    for xb, yb in ds:
        y_prob.append(modelo.predict(xb, verbose=0))
        y_true.append(yb.numpy())
    y_true = np.concatenate(y_true)
    y_prob = np.concatenate(y_prob)
    y_pred = y_prob.argmax(axis=1)
    return y_true, y_pred, y_prob


def reporte_clasificacion(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list[str]) -> dict:
    """precision/recall/F1/support por clase + accuracy + promedios (dict)."""
    return classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0)


def matriz_confusion(y_true: np.ndarray, y_pred: np.ndarray,
                     num_clases: int) -> np.ndarray:
    """Matriz de confusion (filas=verdad, columnas=prediccion)."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_clases)))


def params_entrenables(modelo: tf.keras.Model) -> int:
    """Numero de parametros entrenables (para la tabla comparativa)."""
    return int(sum(np.prod(w.shape) for w in modelo.trainable_weights))


def fila_comparativa(nombre: str, y_true: np.ndarray, y_pred: np.ndarray,
                     *, tiempo_s: float, modelo: tf.keras.Model) -> dict:
    """Una fila de la tabla unificada CNN vs MobileNetV2 Fase 1 vs Fase 2."""
    return {
        "modelo": nombre,
        "accuracy": float((y_true == y_pred).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "tiempo_entren_s": round(float(tiempo_s), 1),
        "params_entrenables": params_entrenables(modelo),
    }


def tabla_comparativa(filas: list[dict]) -> pd.DataFrame:
    """Ensambla las filas en un DataFrame ordenado para informe/notebook."""
    df = pd.DataFrame(filas)
    cols = ["modelo", "accuracy", "f1_macro", "tiempo_entren_s",
            "params_entrenables"]
    return df[cols]


def clase_menor_recall(reporte: dict, class_names: list[str]) -> tuple[str, float]:
    """Clase con menor Recall (responde la pregunta tecnica 18 del informe)."""
    recalls = {c: reporte[c]["recall"] for c in class_names}
    clase = min(recalls, key=recalls.get)
    return clase, recalls[clase]
