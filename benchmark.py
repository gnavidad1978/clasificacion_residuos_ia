"""Costo de computo de cada arquitectura, sin volver a entrenar.

El tiempo total de entrenamiento no sirve para comparar modelos en Colab gratuito:
leer y transformar las fotos con 2 procesadores tarda lo mismo para cualquier red, y
cada modelo se detiene en una epoca distinta. Este script mide la velocidad de cada red
con datos sinteticos ya cargados en memoria, de modo que solo cuenta el computo. Usa la GPU
si esta disponible y, si no, el procesador (en Colab gratuito, unos 45 minutos).

Uso:
    python benchmark.py                       # todos los modelos
    python benchmark.py --modelos deit_small  # solo uno; conserva lo ya medido
Salida:
    resultados/costo_computo.csv
"""

import argparse
import os

import pandas as pd
import torch

from model import crear_red
from utils import contar_parametros, medir_rendimiento

MODELOS = ["mobilenet_v3_large", "resnet50", "yolo11n_cls", "deit_small", "vit_b_16"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=MODELOS)
    ap.add_argument("--lote", type=int, default=32)
    ap.add_argument("--pasos", type=int, default=20)
    ap.add_argument("--fotos_entrenamiento", type=int, default=3105,
                    help="fotos por epoca, para estimar segundos de computo por epoca")
    ap.add_argument("--resultados", default="resultados")
    a = ap.parse_args()

    filas = []
    for nombre in a.modelos:
        torch.manual_seed(0)
        red = crear_red(nombre, 6, pretrained=False)  # los pesos no cambian la velocidad
        r = medir_rendimiento(red, lote=a.lote, pasos=a.pasos)
        r.update({"modelo": nombre, "parametros_millones": contar_parametros(red) / 1e6,
                  "segundos_computo_por_epoca": a.fotos_entrenamiento / r["fotos_por_s_entrenamiento"]})
        filas.append(r)
        print(f"{nombre}: {r['fotos_por_s_entrenamiento']:.0f} fotos/s al entrenar, "
              f"{r['fotos_por_s_inferencia']:.0f} fotos/s al inferir ({r['dispositivo']})")
        del red
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tabla = pd.DataFrame(filas).set_index("modelo")[
        ["parametros_millones", "fotos_por_s_entrenamiento", "fotos_por_s_inferencia",
         "segundos_computo_por_epoca", "memoria_GPU_MB", "dispositivo"]]
    os.makedirs(a.resultados, exist_ok=True)
    ruta = os.path.join(a.resultados, "costo_computo.csv")
    if os.path.exists(ruta):  # conserva las mediciones previas de otros modelos
        previa = pd.read_csv(ruta, index_col="modelo")
        tabla = pd.concat([previa.drop(index=[m for m in tabla.index if m in previa.index]), tabla])
    tabla.to_csv(ruta)
    print("\n" + tabla.round(1).to_string())


if __name__ == "__main__":
    main()
