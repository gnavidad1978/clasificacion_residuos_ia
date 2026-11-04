"""Evaluacion final: cada corrida entrenada contra el conjunto de prueba.

Se ejecuta UNA SOLA VEZ, al final del proyecto, cuando ya no se cambian modelos ni
parametros. Usa el mejor modelo guardado de cada corrida (el checkpoint registrado en
resultados/<modelo>_semilla<N>.json) y reporta media y desviacion entre semillas.
Para evitar repetirla por accidente, si ya existen resultados se niega a correr de nuevo.

Uso:
    python evaluate_test.py                       # conjunto de prueba de RealWaste
    python evaluate_test.py --etiqueta campus --prueba_csv campus.csv --raiz_prueba fotos/
                                                  # otro conjunto ajeno (columnas ruta, categoria)
Salidas en resultados/<etiqueta>/:
    <modelo>_semilla<N>.json, tabla_<etiqueta>.csv, f1_por_categoria_<etiqueta>.csv,
    confusion_<modelo>.png
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, classification_report,
                             confusion_matrix, f1_score)

from categorias import CATEGORIAS, ETIQUETAS
from dataset import ResiduosDataModule
from model import ResiduosModule
from utils import predecir


def media_desv(s, dec=3):
    return f"{s.mean():.{dec}f}" if len(s) == 1 else f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--etiqueta", default="prueba", help="nombre de la carpeta de salida")
    ap.add_argument("--particion_csv", default="data/particion_semilla42.csv")
    ap.add_argument("--raiz_imagenes", default="data/realwaste")
    ap.add_argument("--prueba_csv", default=None, help="otro conjunto (columnas ruta y categoria)")
    ap.add_argument("--raiz_prueba", default=None)
    ap.add_argument("--forzar", action="store_true", help="repetir aunque ya exista (no recomendado)")
    a = ap.parse_args()

    salida = os.path.join(a.resultados, a.etiqueta)
    if os.path.exists(salida) and os.listdir(salida) and not a.forzar:
        print(f"Ya existe {salida}: la evaluacion de '{a.etiqueta}' se hizo antes y no se repite.")
        print("Se muestran los resultados guardados.")
    else:
        os.makedirs(salida, exist_ok=True)
        datos = ResiduosDataModule(particion_csv=a.particion_csv, raiz_imagenes=a.raiz_imagenes,
                                   prueba_csv=a.prueba_csv, raiz_prueba=a.raiz_prueba, batch_size=64)
        datos.setup("test")
        cargador = datos.test_dataloader()
        dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
        archivos = [f for f in sorted(glob.glob(os.path.join(a.resultados, "*.json")))
                    if not f.endswith("linea_base.json")]
        for archivo in archivos:
            r = json.load(open(archivo))
            modelo = ResiduosModule.load_from_checkpoint(r["checkpoint"], map_location="cpu")
            reales, predichas = predecir(modelo, cargador, dispositivo)
            res = {"modelo": r["modelo"], "semilla": r["semilla"], "checkpoint": r["checkpoint"],
                   f"f1_macro_{a.etiqueta}": f1_score(reales, predichas, average="macro",
                                                      labels=list(range(len(CATEGORIAS))), zero_division=0),
                   f"exactitud_{a.etiqueta}": accuracy_score(reales, predichas),
                   "reales": reales, "predichas": predichas}
            nombre = f"{r['modelo']}_semilla{r['semilla']}"
            json.dump(res, open(os.path.join(salida, f"{nombre}.json"), "w"), indent=1)
            print(f"{nombre}: F1 macro {res[f'f1_macro_{a.etiqueta}']:.3f}")
            del modelo

    res = pd.DataFrame([json.load(open(f)) for f in sorted(glob.glob(os.path.join(salida, "*_semilla*.json")))])
    f1, ex = f"f1_macro_{a.etiqueta}", f"exactitud_{a.etiqueta}"
    tabla = res.groupby("modelo").agg(semillas=("semilla", "count"), f1_macro=(f1, media_desv),
                                      exactitud=(ex, media_desv)).sort_values("f1_macro", ascending=False)
    print(f"\nResultados en '{a.etiqueta}' (media ± desviacion entre semillas):\n{tabla.to_string()}")
    tabla.to_csv(os.path.join(salida, f"tabla_{a.etiqueta}.csv"))

    filas = []
    for _, r in res.iterrows():
        rep = classification_report(r["reales"], r["predichas"], labels=list(range(len(CATEGORIAS))),
                                    target_names=ETIQUETAS, output_dict=True, zero_division=0)
        filas += [{"modelo": r["modelo"], "categoria": e, "f1": rep[e]["f1-score"]} for e in ETIQUETAS]
    por_cat = pd.DataFrame(filas).groupby(["categoria", "modelo"])["f1"].agg(media_desv).unstack("modelo")
    por_cat = por_cat.reindex(ETIQUETAS)
    print(f"\nF1 por categoria en '{a.etiqueta}':\n{por_cat.to_string()}")
    por_cat.to_csv(os.path.join(salida, f"f1_por_categoria_{a.etiqueta}.csv"))

    una = (res.assign(orden=(res["semilla"] != 42).astype(int)).sort_values(["orden", "semilla"])
           .groupby("modelo").head(1))
    for _, r in una.iterrows():
        cm = confusion_matrix(r["reales"], r["predichas"], labels=list(range(len(CATEGORIAS))))
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=ETIQUETAS).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                  xticks_rotation=35)
        ax.set_title(f"{r['modelo']} ({a.etiqueta}, semilla {r['semilla']})")
        plt.tight_layout()
        plt.savefig(os.path.join(salida, f"confusion_{r['modelo']}.png"), dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
