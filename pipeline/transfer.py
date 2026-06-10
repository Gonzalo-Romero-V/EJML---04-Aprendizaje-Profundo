"""Transfer Learning con MobileNetV2 (Parte C).

Dos fases sobre MobileNetV2 preentrenado en ImageNet:
  Fase 1 (Feature Extraction): base congelada, se entrena solo la cabeza nueva
    (lr alto, 1e-3) durante 10 epocas.
  Fase 2 (Fine-Tuning): se descongelan las ultimas TL_FT_UNFREEZE_LAST capas y
    se reentrena con lr muy bajo (1e-5) 10 epocas mas, para ajustar suavemente
    sin destruir los pesos preentrenados (pregunta tecnica 14).

Normalizacion: MobileNetV2 espera entradas en [-1, 1]; se aplica su
preprocess_input dentro del modelo (el dataset llega en [0, 255]).

Las BatchNormalization de la base se mantienen congeladas tambien en fine-tuning
(modo inferencia), practica recomendada por Keras para no desestabilizar las
estadisticas preentrenadas (pregunta tecnica 16).
"""
from __future__ import annotations

import tensorflow as tf

from pipeline import config

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
ULTIMO_CONV = "out_relu"          # ultima conv de MobileNetV2 (Grad-CAM opcional)


def construir_transfer(
    *,
    input_shape: tuple[int, int, int] = config.IMG_SHAPE,
    num_clases: int = config.NUM_CLASES,
) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Arma el modelo de Transfer Learning. Devuelve (modelo, base).

    Queda en estado Fase 1: base congelada, cabeza entrenable. Aun sin compilar
    (lo hace compilar_fase1).
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet")
    base.trainable = False                       # Fase 1: feature extraction

    entradas = tf.keras.Input(shape=input_shape, name="entrada")
    x = preprocess_input(entradas)               # [0,255] -> [-1,1]
    x = base(x, training=False)                  # BN en modo inferencia
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="drop_cabeza")(x)
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax",
                                    name="salida")(x)
    modelo = tf.keras.Model(entradas, salidas, name="mobilenetv2_transfer")
    return modelo, base


def compilar_fase1(modelo: tf.keras.Model,
                   lr: float = config.TL_FE_LR) -> tf.keras.Model:
    """Compila para Fase 1 (Feature Extraction)."""
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def preparar_fase2(modelo: tf.keras.Model, base: tf.keras.Model, *,
                   lr: float = config.TL_FT_LR,
                   descongelar_ultimas: int = config.TL_FT_UNFREEZE_LAST
                   ) -> tf.keras.Model:
    """Descongela las ultimas N capas de la base (BN siempre congeladas) y
    recompila con learning rate bajo para el Fine-Tuning."""
    base.trainable = True
    for capa in base.layers[:-descongelar_ultimas]:
        capa.trainable = False
    # Las BatchNorm quedan congeladas aun dentro del bloque descongelado.
    for capa in base.layers[-descongelar_ultimas:]:
        if isinstance(capa, tf.keras.layers.BatchNormalization):
            capa.trainable = False
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def callbacks_entrenamiento(
    patience: int = config.CNN_EARLY_STOPPING_PATIENCE,
) -> list[tf.keras.callbacks.Callback]:
    """Early Stopping sobre val_loss (mismo criterio que la CNN)."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
