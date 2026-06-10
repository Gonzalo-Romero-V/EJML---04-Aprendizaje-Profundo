"""CNN propia desde cero (Parte B).

Arquitectura con 3 bloques convolucionales (Conv-BN-Conv-BN-Pool-Dropout),
cabeza con Global Average Pooling y clasificador softmax de 5 clases. Cada
decision (filtros, kernel, BatchNormalization, Dropout) se justifica en el
informe; los valores viven en config.py y aqui se ensamblan.

Se usa la API funcional para poder exponer la ultima capa convolucional por
nombre (config: 'ultimo_conv'), que Grad-CAM necesita en la Parte B.5.

Normalizacion: esta CNN incorpora Rescaling 1/255 como primera capa (espera
imagenes en [0,1]); el dataset llega en [0,255]. MobileNetV2 (Parte C) usa otra
normalizacion, por eso no se hace en data.py.
"""
from __future__ import annotations

import tensorflow as tf

from pipeline import config

ULTIMO_CONV = "ultimo_conv"          # nombre de la ultima Conv2D (para Grad-CAM)

# Filtros por bloque: se duplican al reducir resolucion (mas mapas para
# representar patrones cada vez mas compuestos con menos detalle espacial).
FILTROS = (32, 64, 128)
KERNEL = 3                            # 3x3: receptivo pequeno, estandar y eficiente
DROPOUT_BLOQUE = 0.25
DROPOUT_CABEZA = 0.5
DENSE_UNITS = 128


def _bloque_conv(x, filtros: int, indice: int, *, ultimo: bool, reg):
    """Bloque Conv-BN-Conv-BN-MaxPool-Dropout."""
    nombre_2da = ULTIMO_CONV if ultimo else f"conv{indice}_2"
    x = tf.keras.layers.Conv2D(filtros, KERNEL, padding="same", activation="relu",
                               kernel_regularizer=reg, name=f"conv{indice}_1")(x)
    x = tf.keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM,
                                           name=f"bn{indice}_1")(x)
    x = tf.keras.layers.Conv2D(filtros, KERNEL, padding="same", activation="relu",
                               kernel_regularizer=reg, name=nombre_2da)(x)
    x = tf.keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM,
                                           name=f"bn{indice}_2")(x)
    x = tf.keras.layers.MaxPooling2D(name=f"pool{indice}")(x)
    x = tf.keras.layers.Dropout(DROPOUT_BLOQUE, name=f"drop{indice}")(x)
    return x


def construir_cnn(
    *,
    input_shape: tuple[int, int, int] = config.IMG_SHAPE,
    num_clases: int = config.NUM_CLASES,
    lr: float = config.CNN_LR,
    l2_reg: float = 0.0,
) -> tf.keras.Model:
    """Devuelve la CNN propia ya compilada (Adam + SparseCategoricalCrossentropy).

    l2_reg > 0 activa regularizacion L2 en las capas conv y densa (lo usa la
    variante mejorada de la Parte D); la CNN base usa 0.0.
    """
    reg = tf.keras.regularizers.l2(l2_reg) if l2_reg else None
    entradas = tf.keras.Input(shape=input_shape, name="entrada")
    x = tf.keras.layers.Rescaling(1.0 / 255, name="rescaling")(entradas)

    n_bloques = len(FILTROS)
    for i, f in enumerate(FILTROS, start=1):
        x = _bloque_conv(x, f, i, ultimo=(i == n_bloques), reg=reg)

    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dense(DENSE_UNITS, activation="relu",
                              kernel_regularizer=reg, name="densa")(x)
    x = tf.keras.layers.BatchNormalization(momentum=config.BN_MOMENTUM,
                                           name="bn_densa")(x)
    x = tf.keras.layers.Dropout(DROPOUT_CABEZA, name="drop_cabeza")(x)
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax", name="salida")(x)

    modelo = tf.keras.Model(entradas, salidas, name="cnn_propia")
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def callbacks_entrenamiento(
    patience: int = config.CNN_EARLY_STOPPING_PATIENCE,
) -> list[tf.keras.callbacks.Callback]:
    """Early Stopping sobre val_loss, restaurando los mejores pesos."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True),
    ]
