"""Datos del proyecto: lee la particion generada por prepare_data.py.

A diferencia de partir las fotos al azar una por una, aqui se usa la particion
guardada en un CSV, hecha por grupos de fotos casi identicas y estratificada por
categoria. Asi una foto y su casi gemela nunca quedan una en entrenamiento y otra
en validacion o prueba, y la particion es la misma en todas las corridas.
"""

import os
from typing import List, Optional

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import lightning as L

from categorias import CATEGORIAS


class FotosResiduos(Dataset):
    """Lista de fotos con su categoria."""

    def __init__(self, rutas: List[str], etiquetas: List[int], transform):
        self.rutas = rutas
        self.etiquetas = etiquetas
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rutas)

    def __getitem__(self, i: int):
        imagen = Image.open(self.rutas[i]).convert("RGB")
        return self.transform(imagen), self.etiquetas[i]


def construir_transformaciones(img_size, mean, std, aumentar, giro_grados=15,
                               brillo=0.3, contraste=0.3, saturacion=0.2):
    """Redimensiona, normaliza y, solo en entrenamiento, aplica aumentos.

    No se usa volteo vertical: un residuo sostenido en la mano no aparece de
    cabeza en el punto de deposito.
    """
    pasos = [transforms.Resize((img_size, img_size))]
    if aumentar:
        pasos += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(giro_grados),
            transforms.ColorJitter(brightness=brillo, contrast=contraste, saturation=saturacion),
        ]
    pasos += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(pasos)


class ResiduosDataModule(L.LightningDataModule):
    """LightningDataModule con la particion fija de entrenamiento, validacion y prueba.

    Args:
        particion_csv:  CSV de prepare_data.py con columnas ruta, categoria, grupo, particion.
        raiz_imagenes:  Carpeta a la que son relativas las rutas del CSV.
        prueba_csv:     Opcional. Otro CSV (columnas ruta y categoria) para evaluar con
                        test, por ejemplo las fotos del campus. Si se da, sustituye a la
                        prueba de la particion.
        raiz_prueba:    Carpeta a la que son relativas las rutas de prueba_csv.
        batch_size:     Tamano de lote.
        img_size:       Lado de la imagen cuadrada.
        mean, std:      Normalizacion por canal (valores publicados de ImageNet, que no
                        se calculan con nuestros datos y por eso no filtran informacion).
        num_workers:    Procesos para leer fotos.
        giro_grados, brillo, contraste, saturacion: intensidad de los aumentos.
    """

    def __init__(
        self,
        particion_csv: str = "data/particion_semilla42.csv",
        raiz_imagenes: str = "data/realwaste",
        prueba_csv: Optional[str] = None,
        raiz_prueba: Optional[str] = None,
        batch_size: int = 32,
        img_size: int = 224,
        mean: List[float] = (0.485, 0.456, 0.406),
        std: List[float] = (0.229, 0.224, 0.225),
        num_workers: int = 2,
        giro_grados: float = 15,
        brillo: float = 0.3,
        contraste: float = 0.3,
        saturacion: float = 0.2,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.ids = {c: i for i, c in enumerate(CATEGORIAS)}

    def _lista(self, tabla, raiz, transform):
        rutas = [os.path.join(raiz, r) for r in tabla["ruta"]]
        etiquetas = [self.ids[c] for c in tabla["categoria"]]
        return FotosResiduos(rutas, etiquetas, transform)

    def setup(self, stage: Optional[str] = None):
        h = self.hparams
        tabla = pd.read_csv(h.particion_csv)
        assert tabla.groupby("grupo")["particion"].nunique().max() == 1, "Hay fuga entre particiones"

        t_ent = construir_transformaciones(h.img_size, h.mean, h.std, True, h.giro_grados,
                                           h.brillo, h.contraste, h.saturacion)
        t_eval = construir_transformaciones(h.img_size, h.mean, h.std, False)

        self.tabla_val = tabla[tabla["particion"] == "validacion"].reset_index(drop=True)
        self.train_ds = self._lista(tabla[tabla["particion"] == "entrenamiento"], h.raiz_imagenes, t_ent)
        self.val_ds = self._lista(self.tabla_val, h.raiz_imagenes, t_eval)
        if h.prueba_csv:
            self.tabla_prueba = pd.read_csv(h.prueba_csv)
            self.test_ds = self._lista(self.tabla_prueba, h.raiz_prueba or "", t_eval)
        else:
            self.tabla_prueba = tabla[tabla["particion"] == "prueba"].reset_index(drop=True)
            self.test_ds = self._lista(self.tabla_prueba, h.raiz_imagenes, t_eval)

        print(f"Particion: entrenamiento {len(self.train_ds)} | validacion {len(self.val_ds)} | "
              f"prueba {len(self.test_ds)}")

    def _cargador(self, ds, mezclar):
        return DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=mezclar,
                          num_workers=self.hparams.num_workers, pin_memory=True,
                          persistent_workers=self.hparams.num_workers > 0)

    def train_dataloader(self):
        return self._cargador(self.train_ds, True)

    def val_dataloader(self):
        return self._cargador(self.val_ds, False)

    def test_dataloader(self):
        return self._cargador(self.test_ds, False)
