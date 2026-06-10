"""Carga del dataset y data augmentation (Parte B.1).

Carga las imagenes de data/dataset/<clase>/ con
tf.keras.utils.image_dataset_from_directory, hace el split 80/20 train/test
exigido por el enunciado y aplica data augmentation SOLO al conjunto de train
(flip horizontal, rotacion, zoom). Devuelve tf.data.Dataset listos para Keras.

Decisiones de diseno:
  - La augmentation se aplica como .map sobre el tf.data de TRAIN unicamente;
    val/test quedan intactos (responde la pregunta tecnica 17 del informe).
  - Las imagenes se devuelven en rango [0, 255]; la normalizacion vive en cada
    modelo (la CNN usa Rescaling 1/255; MobileNetV2 usa su preprocess_input
    a [-1, 1]), porque cada arquitectura espera una escala distinta.
  - Las etiquetas son enteras (label_mode='int') -> perdida
    SparseCategoricalCrossentropy.
"""
from __future__ import annotations

import tensorflow as tf

from pipeline import config

AUTOTUNE = tf.data.AUTOTUNE


def construir_augmentation() -> tf.keras.Sequential:
    """Capa de data augmentation (geometrica, preserva el rango [0,255])."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(config.AUG_FLIP),
            tf.keras.layers.RandomRotation(config.AUG_ROTATION),
            tf.keras.layers.RandomZoom(config.AUG_ZOOM),
        ],
        name="data_augmentation",
    )


def cargar_datasets(
    *,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_STATE,
    augmentar: bool = True,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
    """Devuelve (train_ds, val_ds, class_names) con el split 80/20.

    Con augmentar=True (defecto) la augmentation se mapea SOLO sobre train.
    Ambos datasets quedan cacheados y con prefetch.
    """
    comun = dict(
        directory=config.DATASET_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        validation_split=val_split,
        seed=seed,
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(subset="training", **comun)
    val_ds = tf.keras.utils.image_dataset_from_directory(subset="validation", **comun)
    class_names = train_ds.class_names

    # Cache ANTES de augmentar: se cachean las imagenes crudas y la augmentation
    # se re-aleatoriza en cada epoca (si se cachea despues, queda congelada).
    train_ds = train_ds.cache()
    if augmentar:
        aug = construir_augmentation()
        # training=True para que la augmentation este activa al mapear el train.
        train_ds = train_ds.map(
            lambda x, y: (aug(x, training=True), y), num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    return train_ds, val_ds, class_names


def cargar_test_crudo(
    *,
    img_size: tuple[int, int] = config.IMG_SIZE,
    batch_size: int = config.BATCH_SIZE,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_STATE,
) -> tuple[tf.data.Dataset, list[str]]:
    """Conjunto de validacion/test SIN augmentation, mismo split que el train.

    Util para matriz de confusion, metricas por clase, Grad-CAM e imagenes mal
    clasificadas. Importante: shuffle=True (con el mismo seed que cargar_datasets)
    es OBLIGATORIO para que el split 80/20 quede balanceado entre clases; con
    shuffle=False, image_dataset_from_directory toma como validacion la cola
    alfabetica de archivos (una sola clase). El .cache() fija el orden tras la
    primera pasada, de modo que Grad-CAM y el analisis de errores ven la misma
    secuencia de imagenes.
    """
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=config.DATASET_DIR,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
        validation_split=val_split,
        subset="validation",
        seed=seed,
        shuffle=True,
    )
    class_names = test_ds.class_names
    return test_ds.cache().prefetch(AUTOTUNE), class_names
