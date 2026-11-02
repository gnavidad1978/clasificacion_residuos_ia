"""Reporte comparativo: junta los resultados de todas las corridas.

Lee resultados/*.json (los escribe el callback RegistroResultados) y genera:
    resultados/tabla_comparativa_validacion.csv   media y desviacion por modelo
    resultados/tabla_f1_por_categoria.csv         F1 de cada categoria, media y desviacion
    resultados/figuras/curvas.png                 perdida y F1 por epoca
    resultados/figuras/confusion_<modelo>.png     matriz de confusion en validacion
    resultados/figuras/errores_<modelo>.png       fotos de validacion mal clasificadas

Uso:
    python report.py
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from model import CATEGORIAS


def media_desv(s, dec=3):
    return f"{s.mean():.{dec}f}" if len(s) == 1 else f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--particion_csv", default="data/particion_semilla42.csv")
    ap.add_argument("--raiz_imagenes", default="data/realwaste")
    a = ap.parse_args()

    archivos = [f for f in sorted(glob.glob(os.path.join(a.resultados, "*.json")))
                if not f.endswith("linea_base.json")]
    assert archivos, "No hay resultados; entrena primero con main.py fit"
    res = pd.DataFrame([json.load(open(f)) for f in archivos])
    # Segundos por epoca (incluye la validacion de cada epoca); las corridas antiguas no lo
    # guardaban, se estima con minutos totales entre epocas entrenadas
    if "segundos_por_epoca" not in res:
        res["segundos_por_epoca"] = None
    res["seg_epoca"] = [r["segundos_por_epoca"] if pd.notna(r["segundos_por_epoca"])
                        else 60 * r["minutos_entrenamiento"] / len(r["historia"]) for _, r in res.iterrows()]
    res["epocas"] = [len(h) for h in res["historia"]]
    dir_fig = os.path.join(a.resultados, "figuras")
    os.makedirs(dir_fig, exist_ok=True)

    # Tabla comparativa
    tabla = res.groupby("modelo").agg(
        semillas=("semilla", "count"),
        f1_macro_val=("f1_macro_val", media_desv),
        exactitud_val=("exactitud_val", media_desv),
        parametros_M=("parametros_millones", lambda s: f"{s.iloc[0]:.1f}"),
        tamano_MB=("tamano_MB", lambda s: f"{s.iloc[0]:.0f}"),
        epocas=("epocas", lambda s: media_desv(s, 1)),
        seg_por_epoca=("seg_epoca", lambda s: media_desv(s, 0)),
        ms_por_foto_cpu=("ms_por_foto_cpu", lambda s: media_desv(s, 0)),
    )
    ruta_costo = os.path.join(a.resultados, "costo_computo.csv")
    if os.path.exists(ruta_costo):
        costo = pd.read_csv(ruta_costo, index_col="modelo")
        tabla["fotos_s_entren_GPU"] = costo["fotos_por_s_entrenamiento"].round(0).astype("Int64")
        tabla["fotos_s_inferencia_GPU"] = costo["fotos_por_s_inferencia"].round(0).astype("Int64")
    tabla = tabla.sort_values("f1_macro_val", ascending=False)
    ruta_base = os.path.join(a.resultados, "linea_base.json")
    if os.path.exists(ruta_base):
        base = json.load(open(ruta_base))
        print(f"Linea base (siempre '{base['clase_mayoritaria']}'): F1 macro {base['f1_macro']:.3f}\n")
    print(tabla.to_string())
    tabla.to_csv(os.path.join(a.resultados, "tabla_comparativa_validacion.csv"))

    # F1 por categoria, media y desviacion entre semillas
    filas = []
    for _, r in res.iterrows():
        rep_cat = classification_report(r["reales_val"], r["predichas_val"], labels=list(range(len(CATEGORIAS))),
                                        target_names=CATEGORIAS, output_dict=True, zero_division=0)
        for c in CATEGORIAS:
            filas.append({"modelo": r["modelo"], "categoria": c, "f1": rep_cat[c]["f1-score"]})
    por_cat = pd.DataFrame(filas).groupby(["categoria", "modelo"])["f1"].agg(media_desv).unstack("modelo")
    por_cat = por_cat.reindex(CATEGORIAS)
    print("\nF1 por categoria (validacion, media entre semillas):")
    print(por_cat.to_string())
    por_cat.to_csv(os.path.join(a.resultados, "tabla_f1_por_categoria.csv"))

    # Curvas: una corrida por modelo, de preferencia la semilla 42
    primera = (res.assign(orden=(res["semilla"] != 42).astype(int))
               .sort_values(["orden", "semilla"]).groupby("modelo").head(1).reset_index(drop=True))
    fig, ejes = plt.subplots(1, len(primera), figsize=(5.5 * len(primera), 4.3), squeeze=False)
    for ax, (_, r) in zip(ejes[0], primera.iterrows()):
        h = pd.DataFrame(r["historia"])
        ax.plot(h["epoca"], h["perdida_ent"], label="entrenamiento")
        ax.plot(h["epoca"], h["perdida_val"], label="validacion")
        ax.axvline(r["mejor_epoca"], color="gray", ls=":", lw=1)
        ax2 = ax.twinx()
        ax2.plot(h["epoca"], h["f1_val"], color="green", ls="--", label="F1 val")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("F1 macro validacion", color="green")
        ax.set_title(f"{r['modelo']}, semilla {r['semilla']}")
        ax.set_xlabel("epoca")
        ax.set_ylabel("perdida")
        lineas = ax.get_lines()[:2] + ax2.get_lines()
        ax.legend(lineas, [l.get_label() for l in lineas], loc="upper center",
                  bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_fig, "curvas.png"), dpi=150)
    plt.close()

    # Matriz de confusion, reporte por categoria y errores
    validacion = pd.read_csv(a.particion_csv)
    validacion = validacion[validacion["particion"] == "validacion"].reset_index(drop=True)
    etiquetas = list(range(len(CATEGORIAS)))
    for _, r in primera.iterrows():
        print(f"\n{r['modelo']}, semilla {r['semilla']}")
        print(classification_report(r["reales_val"], r["predichas_val"], labels=etiquetas,
                                    target_names=CATEGORIAS, digits=3, zero_division=0))
        cm = confusion_matrix(r["reales_val"], r["predichas_val"], labels=etiquetas)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=CATEGORIAS).plot(ax=ax, cmap="Blues", colorbar=False,
                                                                   xticks_rotation=35)
        ax.set_title(f"{r['modelo']} (validacion)")
        plt.tight_layout()
        plt.savefig(os.path.join(dir_fig, f"confusion_{r['modelo']}.png"), dpi=150)
        plt.close()

        errores = [i for i, (x, y) in enumerate(zip(r["reales_val"], r["predichas_val"])) if x != y][:12]
        if errores and len(validacion) == len(r["reales_val"]):
            col = 6
            filas = (len(errores) + col - 1) // col
            fig, ejes = plt.subplots(filas, col, figsize=(2.6 * col, 2.9 * filas), squeeze=False)
            for ax in ejes.flat:
                ax.axis("off")
            for ax, i in zip(ejes.flat, errores):
                ruta = validacion.loc[i, "ruta"]
                ax.imshow(Image.open(os.path.join(a.raiz_imagenes, ruta)).convert("RGB"))
                ax.set_title(f"real: {CATEGORIAS[r['reales_val'][i]]}\ncreyo: "
                             f"{CATEGORIAS[r['predichas_val'][i]]}\n{os.path.basename(ruta)}", fontsize=7)
            fig.suptitle(f"Errores de {r['modelo']} en validacion", fontsize=11)
            plt.tight_layout()
            plt.savefig(os.path.join(dir_fig, f"errores_{r['modelo']}.png"), dpi=150)
            plt.close()
    print(f"\nFiguras guardadas en {dir_fig}")


if __name__ == "__main__":
    main()
