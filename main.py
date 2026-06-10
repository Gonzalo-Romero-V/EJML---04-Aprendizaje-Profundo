"""Punto de entrada del Ejercicio 4 - Aprendizaje Profundo (headless).

Corre el pipeline completo y deja todo listo para el informe y el notebook:

    Parte A  -> dataset ya preparado por scripts/preparar_dataset.py
    Parte B  -> CNN propia: entrenamiento, curvas, metricas, matriz, Grad-CAM
    Parte C  -> Transfer Learning MobileNetV2 (Fase 1 + Fase 2) y comparativa
    Parte D  -> par mas confundido, CNN mejorada (2 mejoras) e imagenes erroneas

Genera las figuras en docs/figuras/ y un resumen en data/resultados.json para
que el notebook y el informe los consuman sin reentrenar. El entrenamiento real
(CNN_EPOCHS=40 + TL 10+10) es GPU-intensivo: pensado para Google Colab. En CPU
local conviene bajar las epocas en pipeline/config.py para una corrida rapida.

    Uso:  python main.py
"""
from __future__ import annotations

import json
import sys
import time

for _stream in (sys.stdout, sys.stderr):       # UTF-8 en cualquier consola
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

import numpy as np  # noqa: E402

from pipeline import (audit, cnn, config, data, evaluation as ev,  # noqa: E402
                      plotting, transfer)


def _entrenar(modelo, tr, va, epochs, callbacks):
    t0 = time.time()
    h = modelo.fit(tr, validation_data=va, epochs=epochs,
                   callbacks=callbacks, verbose=2)
    return h.history, time.time() - t0


def main() -> None:
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    resultados: dict = {}

    print("\n[Datos] Cargando dataset (split 80/20, augmentation solo en train)")
    tr, va, clases = data.cargar_datasets()
    test_ds, _ = data.cargar_test_crudo()
    print(f"  · clases: {clases}")

    # ── Parte B — CNN desde cero ──────────────────────────────────────────────
    print("\n[Parte B] CNN propia")
    modelo_cnn = cnn.construir_cnn()
    print(f"  · parametros: {modelo_cnn.count_params():,}")
    hist_cnn, t_cnn = _entrenar(modelo_cnn, tr, va, config.CNN_EPOCHS,
                                cnn.callbacks_entrenamiento())
    fig_cnn = plotting.curvas_entrenamiento(
        hist_cnn, "cnn_curvas.png", "CNN propia — entrenamiento")
    yt, yp, _ = ev.predecir(modelo_cnn, test_ds)
    rep_cnn = ev.reporte_clasificacion(yt, yp, clases)
    cm_cnn = ev.matriz_confusion(yt, yp, len(clases))
    fig_cm_cnn = plotting.matriz_confusion_fig(
        cm_cnn, clases, "confusion_cnn.png", "Matriz de confusion — CNN propia")
    print(f"  · accuracy {rep_cnn['accuracy']:.3f} | "
          f"F1 macro {rep_cnn['macro avg']['f1-score']:.3f}")

    # Grad-CAM sobre 3 imagenes de test bien clasificadas (mapas mas legibles).
    imgs, yt_g, yp_g = audit.recolectar_predicciones(modelo_cnn, test_ds)
    aciertos = np.flatnonzero(yt_g == yp_g)[:3]
    fig_gradcam = plotting.grid_gradcam(
        modelo_cnn, imgs[aciertos], clases, yt_g[aciertos], yp_g[aciertos],
        "gradcam_cnn.png")

    # ── Parte C — Transfer Learning MobileNetV2 ──────────────────────────────
    print("\n[Parte C] Transfer Learning (MobileNetV2)")
    modelo_tl, base = transfer.construir_transfer()
    transfer.compilar_fase1(modelo_tl)
    print(f"  · Fase 1 (feature extraction): "
          f"{ev.params_entrenables(modelo_tl):,} params entrenables")
    hist_f1, t_f1 = _entrenar(modelo_tl, tr, va, config.TL_FE_EPOCHS,
                              transfer.callbacks_entrenamiento())
    yt1, yp1, _ = ev.predecir(modelo_tl, test_ds)
    rep_f1 = ev.reporte_clasificacion(yt1, yp1, clases)

    transfer.preparar_fase2(modelo_tl, base)
    print(f"  · Fase 2 (fine-tuning): "
          f"{ev.params_entrenables(modelo_tl):,} params entrenables")
    hist_f2, t_f2 = _entrenar(modelo_tl, tr, va, config.TL_FT_EPOCHS,
                              transfer.callbacks_entrenamiento())
    yt2, yp2, _ = ev.predecir(modelo_tl, test_ds)
    rep_f2 = ev.reporte_clasificacion(yt2, yp2, clases)
    cm_f2 = ev.matriz_confusion(yt2, yp2, len(clases))
    plotting.matriz_confusion_fig(
        cm_f2, clases, "confusion_tl.png",
        "Matriz de confusion — MobileNetV2 Fase 2")

    # Curvas TL: Fase 1 + Fase 2 encadenadas, con vertical en el cambio de fase.
    hist_tl = {k: hist_f1[k] + hist_f2.get(k, []) for k in hist_f1}
    plotting.curvas_entrenamiento(
        hist_tl, "tl_curvas.png", "MobileNetV2 — Fase 1 + Fase 2",
        linea_finetuning=len(hist_f1["accuracy"]))
    print(f"  · F1: acc {rep_f1['accuracy']:.3f} | "
          f"Fase 2: acc {rep_f2['accuracy']:.3f}")

    # ── Tabla comparativa de los 3 modelos ───────────────────────────────────
    filas = [
        ev.fila_comparativa("CNN propia", yt, yp, tiempo_s=t_cnn,
                            modelo=modelo_cnn),
        ev.fila_comparativa("MobileNetV2 Fase 1", yt1, yp1, tiempo_s=t_f1,
                            modelo=modelo_tl),
        ev.fila_comparativa("MobileNetV2 Fase 2", yt2, yp2, tiempo_s=t_f2,
                            modelo=modelo_tl),
    ]
    df = ev.tabla_comparativa(filas)
    plotting.tabla_comparativa_fig(df, "tabla_comparativa.png")
    print("\n[Comparativa]\n" + df.to_string(index=False))

    # ── Parte D — Auditoria y mejora ─────────────────────────────────────────
    print("\n[Parte D] Auditoria")
    ca, cb, n_conf = audit.par_mas_confundido(cm_cnn, clases)
    print(f"  · par mas confundido (CNN): {ca} <-> {cb} ({n_conf} casos)")

    f1_por_modelo = {f["modelo"]: f["f1_macro"] for f in filas}
    peor = min(f1_por_modelo, key=f1_por_modelo.get)
    print(f"  · modelo con menor F1: {peor} ({f1_por_modelo[peor]:.3f})")
    # La mejora se aplica sobre la CNN propia (arquitectura propia, la mas
    # mejorable; las mejoras 1/2 son L2 + ReduceLROnPlateau).
    print("  · reentrenando CNN mejorada (L2 + ReduceLROnPlateau)…")
    modelo_mej = audit.construir_cnn_mejorada()
    _, t_mej = _entrenar(modelo_mej, tr, va, config.CNN_EPOCHS,
                         audit.callbacks_mejorados())
    ytm, ypm, _ = ev.predecir(modelo_mej, test_ds)
    f1_mej = ev.reporte_clasificacion(ytm, ypm, clases)["macro avg"]["f1-score"]
    f1_cnn = rep_cnn["macro avg"]["f1-score"]
    mejoro = f1_mej > f1_cnn
    print(f"  · F1 CNN base {f1_cnn:.3f} -> mejorada {f1_mej:.3f} "
          f"({'mejoro' if mejoro else 'no mejoro'})")

    # Mejor modelo global (mayor F1 macro) -> imagenes mal clasificadas.
    mejor_nombre = max(f1_por_modelo, key=f1_por_modelo.get)
    usar = (modelo_cnn, yt, yp) if mejor_nombre == "CNN propia" else \
           (modelo_tl, yt2, yp2)
    imgs_b, ytb, ypb = audit.recolectar_predicciones(usar[0], test_ds)
    err = audit.indices_mal_clasificados(ytb, ypb, maximo=8)
    titulos = [f"{clases[ytb[i]]}->{clases[ypb[i]]}" for i in err]
    plotting.grid_imagenes(imgs_b[err], titulos, "errores_mejor_modelo.png",
                           f"Mal clasificadas — {mejor_nombre}")
    print(f"  · mejor modelo: {mejor_nombre} | imagenes mal clasificadas: "
          f"{len(np.flatnonzero(ytb != ypb))}")

    # ── Guardar modelos y resumen ────────────────────────────────────────────
    modelo_cnn.save(config.MODELS_DIR / "cnn_propia.keras")
    modelo_tl.save(config.MODELS_DIR / "mobilenetv2_tl.keras")
    resultados.update({
        "clases": clases,
        "cnn": {"report": rep_cnn, "cm": cm_cnn.tolist(),
                "epocas": len(hist_cnn["accuracy"]), "tiempo_s": round(t_cnn, 1),
                "params": modelo_cnn.count_params()},
        "tl_fase1": {"report": rep_f1, "tiempo_s": round(t_f1, 1)},
        "tl_fase2": {"report": rep_f2, "cm": cm_f2.tolist(),
                     "tiempo_s": round(t_f2, 1)},
        "comparativa": filas,
        "menor_recall_cnn": ev.clase_menor_recall(rep_cnn, clases),
        "parte_d": {
            "par_confundido": [ca, cb, n_conf],
            "peor_modelo": peor,
            "mejora": {"f1_base": round(f1_cnn, 4),
                       "f1_mejorada": round(f1_mej, 4), "mejoro": bool(mejoro),
                       "tiempo_s": round(t_mej, 1)},
            "mejor_modelo": mejor_nombre,
        },
    })
    with open(config.RESULTADOS_JSON, "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Pipeline completado. Resumen -> "
          f"{config.RESULTADOS_JSON.relative_to(config.ROOT)} | figuras en "
          f"{config.FIGURES_DIR.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
