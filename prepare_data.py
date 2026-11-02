"""Prepara RealWaste: descarga, acomodo a las seis categorias, particion y exploracion.

Equivale al create_data.sh del repositorio de la clase, mas el protocolo experimental:
semilla fija, particion antes de mirar los datos, exploracion solo del entrenamiento
y linea base. Es determinista: con la misma semilla produce siempre la misma particion.

Uso:
    python prepare_data.py --zip /content/drive/MyDrive/proyecto_IA/realwaste.zip

Salidas:
    data/realwaste/                       fotos descomprimidas
    data/particion_semilla42.csv          particion (rutas relativas a data/realwaste)
    resultados/linea_base.json            F1 macro de la clase mayoritaria en validacion
    resultados/eda/*.png                  figuras de la exploracion

Fuente de los datos: S. Single, S. Iranmanesh y R. Raad, "RealWaste: A Novel Real-Life
Data Set for Landfill Waste Classification Using Deep Learning", Information, vol. 14,
no. 12, p. 633, 2023, doi: 10.3390/info14120633. Conjunto en UCI Machine Learning
Repository, doi: 10.24432/C5SS4G, licencia CC BY 4.0.
"""

import argparse
import json
import os
import urllib.request
import zipfile
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

from model import CATEGORIAS

URL_REALWASTE = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"

# Acomodo de las clases de RealWaste a las categorias de la NAE-SEMADES-007/2008.
# Miscellaneous Trash a dificil reciclaje es interpretacion propia, pendiente de revision
# por muestreo. Textile Trash se excluye: el campus no tiene bote de textiles.
MAPEO = {
    "Cardboard": "papel_carton",
    "Paper": "papel_carton",
    "Glass": "vidrio",
    "Metal": "metal",
    "Plastic": "plastico",
    "Food Organics": "organico",
    "Vegetation": "organico",
    "Miscellaneous Trash": "dificil_reciclaje",
    "Textile Trash": None,
}
EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def descargar_y_descomprimir(ruta_zip, destino):
    if not os.path.exists(ruta_zip):
        print("Descargando RealWaste (unos 657 MB)...")
        os.makedirs(os.path.dirname(ruta_zip) or ".", exist_ok=True)
        urllib.request.urlretrieve(URL_REALWASTE, ruta_zip)
    if not os.path.exists(destino):
        print("Descomprimiendo...")
        with zipfile.ZipFile(ruta_zip) as z:
            z.extractall(destino)


def inventariar(raiz, mapeo):
    filas = []
    for carpeta, subcarpetas, archivos in os.walk(raiz):
        subcarpetas.sort()  # orden fijo, para que el resultado no dependa del disco
        clase = os.path.basename(carpeta)
        if clase not in mapeo:
            continue
        for nombre in sorted(archivos):
            if nombre.lower().endswith(EXTENSIONES):
                filas.append({"ruta": os.path.relpath(os.path.join(carpeta, nombre), raiz),
                              "clase_original": clase, "categoria": mapeo[clase]})
    return pd.DataFrame(filas)


def huella_dhash(ruta, lado):
    # Reduce la foto a (lado+1) x lado en grises y compara cada pixel con su vecino.
    img = Image.open(ruta).convert("L").resize((lado + 1, lado), Image.BILINEAR)
    a = np.asarray(img, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def agrupar_casi_duplicados(df, raiz, umbral, lado):
    grupo = np.arange(len(df))

    def raiz_de(i):
        while grupo[i] != i:
            grupo[i] = grupo[grupo[i]]
            i = grupo[i]
        return i

    for _, sub in df.groupby("categoria"):
        idx = sub.index.to_numpy()
        H = np.stack([huella_dhash(os.path.join(raiz, r), lado) for r in sub["ruta"]])
        for k in range(len(idx)):
            dist = (H[k + 1:] != H[k]).sum(axis=1)
            for j in np.nonzero(dist <= umbral)[0]:
                a, b = raiz_de(idx[k]), raiz_de(idx[k + 1 + j])
                if a != b:
                    grupo[b] = a
    return np.array([raiz_de(i) for i in range(len(df))])


def particionar(df, rng, p_ent, p_val):
    """Asigna grupos completos de modo que cada conjunto reciba cerca de p_ent, p_val y
    el resto de las fotos de cada categoria."""
    nombres = ["entrenamiento", "validacion", "prueba"]
    proporciones = np.array([p_ent, p_val, 1 - p_ent - p_val])
    asignacion = {}
    for _, sub in df.groupby("categoria"):
        tam = sub["grupo"].value_counts()
        grupos = tam.index.to_numpy().copy()
        rng.shuffle(grupos)
        grupos = sorted(grupos, key=lambda g: -tam[g])
        meta, actual = proporciones * len(sub), np.zeros(3)
        for g in grupos:
            k = int(np.argmax(meta - actual))
            asignacion[g] = nombres[k]
            actual[k] += tam[g]
    return df["grupo"].map(asignacion)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", default="data/realwaste.zip", help="ruta del zip de RealWaste")
    ap.add_argument("--destino", default="data/realwaste")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--entrenamiento", type=float, default=0.70)
    ap.add_argument("--validacion", type=float, default=0.15)
    ap.add_argument("--lado-huella", type=int, default=16)
    ap.add_argument("--umbral-casi-duplicado", type=int, default=10)
    ap.add_argument("--excluir-misc", action="store_true",
                    help="excluye Miscellaneous Trash si la revision no sostiene su acomodo")
    ap.add_argument("--resultados", default="resultados")
    a = ap.parse_args()

    rng = np.random.default_rng(a.semilla)  # regla 1: semilla fija
    mapeo = dict(MAPEO)
    if a.excluir_misc:
        mapeo["Miscellaneous Trash"] = None
    dir_eda = os.path.join(a.resultados, "eda")
    os.makedirs(dir_eda, exist_ok=True)

    descargar_y_descomprimir(a.zip, a.destino)

    inventario = inventariar(a.destino, mapeo)
    assert len(inventario) > 0, "No se encontraron fotos"
    print("Fotos por clase original:")
    print(inventario["clase_original"].value_counts().sort_index().to_string())
    datos = inventario.dropna(subset=["categoria"]).reset_index(drop=True)
    print(f"Excluidas por el acomodo: {inventario['categoria'].isna().sum()}. Se usan: {len(datos)}")

    datos["grupo"] = agrupar_casi_duplicados(datos, a.destino, a.umbral_casi_duplicado, a.lado_huella)
    tam = datos["grupo"].value_counts()
    print(f"Grupos: {datos.grupo.nunique()} para {len(datos)} fotos; "
          f"{(tam > 1).sum()} grupos con mas de una foto; grupos mayores: {tam.head(5).tolist()}")

    # regla 2: particion antes de mirar
    datos["particion"] = particionar(datos, rng, a.entrenamiento, a.validacion)
    assert datos.groupby("grupo")["particion"].nunique().max() == 1
    ruta_csv = os.path.join(os.path.dirname(a.destino) or ".", f"particion_semilla{a.semilla}.csv")
    datos.to_csv(ruta_csv, index=False)
    tabla = pd.crosstab(datos["categoria"], datos["particion"], margins=True)[
        ["entrenamiento", "validacion", "prueba", "All"]]
    print(f"\nParticion guardada en {ruta_csv}\n{tabla.to_string()}")
    print("\nPorcentajes:\n" + (tabla.div(tabla["All"], axis=0) * 100).round(1).drop(columns="All").to_string())

    # exploracion solo del entrenamiento
    ent = datos[datos["particion"] == "entrenamiento"]
    val = datos[datos["particion"] == "validacion"]
    conteo = ent["categoria"].value_counts().reindex(CATEGORIAS)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(conteo.index, conteo.values, color="#4C78A8")
    for i, v in enumerate(conteo.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Fotos de entrenamiento")
    ax.set_title("Fotos por categoria en entrenamiento")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_eda, "conteo_entrenamiento.png"), dpi=150)
    plt.close()

    formatos, tamanos, modos = Counter(), Counter(), Counter()
    for r in ent["ruta"]:
        with Image.open(os.path.join(a.destino, r)) as im:
            formatos[im.format] += 1
            tamanos[im.size] += 1
            modos[im.mode] += 1
    print(f"\nEntrenamiento: formatos {dict(formatos)}, modos {dict(modos)}, "
          f"tamanos mas comunes {tamanos.most_common(3)}; "
          f"razon mayor/menor categoria {conteo.max() / conteo.min():.2f}")

    fig, ejes = plt.subplots(len(CATEGORIAS), 4, figsize=(9, 2.3 * len(CATEGORIAS)))
    for f, cat in enumerate(CATEGORIAS):
        rutas = ent.loc[ent["categoria"] == cat, "ruta"].to_numpy()
        elegidas = rng.choice(rutas, size=min(4, len(rutas)), replace=False)
        for c in range(4):
            ejes[f, c].axis("off")
            if c < len(elegidas):
                ejes[f, c].imshow(Image.open(os.path.join(a.destino, elegidas[c])).convert("RGB"))
        ejes[f, 0].set_title(cat, loc="left", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_eda, "muestras_entrenamiento.png"), dpi=120)
    plt.close()

    # regla 3: linea base, evaluada en validacion
    mayoritaria = ent["categoria"].value_counts().idxmax()
    f1_base = f1_score(val["categoria"], [mayoritaria] * len(val), labels=CATEGORIAS,
                       average="macro", zero_division=0)
    exac_base = accuracy_score(val["categoria"], [mayoritaria] * len(val))
    with open(os.path.join(a.resultados, "linea_base.json"), "w") as f:
        json.dump({"clase_mayoritaria": mayoritaria, "f1_macro": f1_base,
                   "exactitud": exac_base, "semilla": a.semilla}, f, indent=2)
    print(f"\nLinea base (siempre '{mayoritaria}') en validacion: F1 macro {f1_base:.3f}, "
          f"exactitud {exac_base:.3f}, azar uniforme {1 / len(CATEGORIAS):.3f}")


if __name__ == "__main__":
    main()
