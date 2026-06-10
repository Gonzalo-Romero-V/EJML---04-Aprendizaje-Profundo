"""Grad-CAM (Parte B.5).

Gradient-weighted Class Activation Mapping: visualiza que zonas de la imagen
activan la red para una clase. Se calcula el gradiente del score de la clase
objetivo respecto a los mapas de la ultima capa convolucional
(cnn.ULTIMO_CONV), se promedian esos gradientes por canal (importancia de cada
mapa) y se combinan los mapas con esos pesos, aplicando ReLU.

Obligatorio en la tarea (no implementarlo: -1). Funciona sobre la CNN propia,
cuya ultima conv esta nombrada; para MobileNetV2 se pasa el nombre de su ultima
conv ('out_relu').
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf

from pipeline import cnn


def heatmap(modelo: tf.keras.Model, imagen: np.ndarray, *,
            capa_conv: str = cnn.ULTIMO_CONV,
            clase_idx: int | None = None) -> tuple[np.ndarray, int]:
    """Mapa de calor Grad-CAM normalizado [0,1] para una imagen.

    `imagen`: array (H, W, 3) en el mismo rango que espera el modelo (la
    normalizacion ocurre dentro del modelo). Devuelve (heatmap, clase_usada).
    """
    x = tf.convert_to_tensor(imagen[None, ...], dtype=tf.float32)
    grad_model = tf.keras.Model(
        modelo.inputs, [modelo.get_layer(capa_conv).output, modelo.output])

    with tf.GradientTape() as tape:
        conv_out, predicciones = grad_model(x, training=False)
        if clase_idx is None:
            clase_idx = int(tf.argmax(predicciones[0]))
        score = predicciones[:, clase_idx]

    grads = tape.gradient(score, conv_out)               # dScore/dActivaciones
    pesos = tf.reduce_mean(grads, axis=(0, 1, 2))        # importancia por canal
    conv_out = conv_out[0]
    mapa = tf.reduce_sum(conv_out * pesos, axis=-1)      # combinacion ponderada
    mapa = tf.nn.relu(mapa)                               # solo influencias +
    maximo = tf.reduce_max(mapa)
    if maximo > 0:
        mapa = mapa / maximo
    return mapa.numpy(), int(clase_idx)


def superponer(imagen: np.ndarray, mapa: np.ndarray, *,
               alpha: float = 0.4) -> np.ndarray:
    """Redimensiona el heatmap al tamano de la imagen y lo superpone (RGB).

    `imagen`: array en [0,255]. Devuelve un array uint8 listo para imshow.
    """
    import matplotlib.cm as cm

    h, w = imagen.shape[:2]
    mapa = tf.image.resize(mapa[..., None], (h, w)).numpy()[..., 0]
    colores = cm.jet(mapa)[..., :3]                       # heatmap a color
    base = imagen.astype("float32") / 255.0
    fusion = (1 - alpha) * base + alpha * colores
    return (np.clip(fusion, 0, 1) * 255).astype("uint8")
