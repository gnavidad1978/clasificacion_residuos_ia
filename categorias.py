"""Categorias del proyecto y constantes compartidas (sin dependencias pesadas).

El identificador interno ``dificil_reciclaje`` se conserva para que la particion sea
identica a la de las corridas ya hechas (la particion recorre las categorias en orden
alfabetico de su identificador). Sin embargo, una revision por muestreo de 50 fotos
mostro que la clase de origen, Miscellaneous Trash, mezcla residuos de muchas clases y
no corresponde a "residuos inorganicos de dificil reciclaje" de la NAE-SEMADES-007/2008.
Por eso en tablas y figuras se muestra como "otros".
"""

CATEGORIAS = ["vidrio", "plastico", "metal", "papel_carton", "dificil_reciclaje", "organico"]

NOMBRES = {
    "vidrio": "vidrio",
    "plastico": "plastico",
    "metal": "metal",
    "papel_carton": "papel_carton",
    "dificil_reciclaje": "otros_misc",
    "organico": "organico",
}
ETIQUETAS = [NOMBRES[c] for c in CATEGORIAS]

MEDIA_IMAGENET = (0.485, 0.456, 0.406)
DESV_IMAGENET = (0.229, 0.224, 0.225)
