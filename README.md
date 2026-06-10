# Aprendizaje Profundo — Clasificación de residuos para reciclaje

Ejercicio de visión por computador con redes neuronales: se clasifican imágenes
de residuos en 5 categorías reciclables comparando **una CNN entrenada desde
cero** contra **Transfer Learning con MobileNetV2**, y se audita el mejor modelo
para proponer mejoras concretas.

## Contenido

- **Parte A — Dataset:** [TrashNet](https://github.com/garythung/trashnet)
  (Stanford CS229), versión redimensionada. `scripts/preparar_dataset.py` lo
  descarga, selecciona 5 clases (`cardboard`, `glass`, `metal`, `paper`,
  `plastic`) y las recorta a 400 imágenes c/u (subsample con semilla fija) →
  dataset balanceado de 2000 imágenes.
- **Parte B — CNN propia:** red convolucional desde cero (≥3 bloques conv con
  BatchNorm y Dropout), data augmentation solo en train (flip, rotación, zoom),
  curvas de entrenamiento, matriz de confusión y **Grad-CAM** para interpretar
  en qué se fija la red.
- **Parte C — Transfer Learning (MobileNetV2):** Fase 1 (feature extraction, base
  congelada) + Fase 2 (fine-tuning de las últimas 30 capas con learning rate
  bajo), y tabla comparativa de los 3 modelos.
- **Parte D — Auditoría y mejora:** par de clases más confundido, CNN mejorada
  (regularización L2 + ReduceLROnPlateau) y galería de imágenes mal clasificadas
  por el mejor modelo.

## Estructura

```
.
├── main.py                       # Orquesta las 4 partes de punta a punta (headless)
├── scripts/
│   └── preparar_dataset.py       # Parte A: descarga y balancea el dataset (solo stdlib)
├── pipeline/
│   ├── config.py                 # Semilla, clases, hiperparámetros y rutas
│   ├── data.py                   # Carga del dataset, split 80/20 y augmentation
│   ├── cnn.py                    # CNN propia desde cero
│   ├── transfer.py               # Transfer Learning con MobileNetV2 (Fase 1 + 2)
│   ├── evaluation.py             # Métricas, matriz de confusión y tabla comparativa
│   ├── gradcam.py                # Mapas de activación Grad-CAM
│   ├── audit.py                  # Parte D: par confundido, CNN mejorada, errores
│   └── plotting.py               # Figuras (curvas, matrices, Grad-CAM, grillas)
├── notebooks/
│   └── informe_profundo.ipynb    # Análisis ejecutable en Jupyter / Colab
├── data/                         # Dataset, modelos y resumen (no versionados)
├── docs/figuras/                 # Figuras generadas por main.py (no versionadas)
└── requirements.txt
```

## Instalación

```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

## Uso

1. **Preparar el dataset** (descarga TrashNet y construye `data/dataset/<clase>/`):

   ```bash
   python scripts/preparar_dataset.py
   ```

2. **Ejecutar el pipeline completo** (entrena los modelos, evalúa, guarda las
   figuras en `docs/figuras/` y un resumen en `data/resultados.json`):

   ```bash
   python main.py
   ```

El entrenamiento real (CNN 40 épocas + fine-tuning de MobileNetV2) es
GPU-intensivo y está pensado para **Google Colab**. En CPU local conviene
reducir las épocas en `pipeline/config.py` para una corrida de validación
rápida. También se puede recorrer paso a paso en
`notebooks/informe_profundo.ipynb` (detecta Colab automáticamente).

## Reproducibilidad

Toda la aleatoriedad está fijada con `RANDOM_STATE = 42` (`pipeline/config.py`):
selección y balanceo del dataset, split 80/20 y augmentation. Las versiones de
las dependencias están ancladas en `requirements.txt`. Los artefactos derivados
—el dataset, los modelos `.keras`, las figuras y el `resultados.json`— no se
versionan: se regeneran ejecutando `scripts/preparar_dataset.py` y `python
main.py`. Sobre GPU el entrenamiento conserva una pequeña varianza no
determinista propia de cuDNN; las métricas reproducen los rangos del informe.
