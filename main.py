"""Punto de entrada principal. Usa LightningCLI, igual que el repositorio de la clase.

Ejemplos de uso (desde la carpeta del repositorio):

    # Entrenar un modelo con su configuracion
    python main.py fit --config configs/mobilenet_v3.yaml

    # Otra semilla, sin tocar el YAML
    python main.py fit --config configs/mobilenet_v3.yaml --seed_everything 7

    # Cambiar cualquier parametro desde la linea de comandos
    python main.py fit --config configs/vit_b_16.yaml --trainer.max_epochs 20 --model.lr 0.00005

    # Evaluacion final con el conjunto de prueba (una sola vez, al final del proyecto)
    python main.py test --config configs/mobilenet_v3.yaml --ckpt_path runs/mobilenet_v3_large/version_0/checkpoints/<archivo>.ckpt
"""

import torch
from lightning.pytorch.cli import LightningCLI

from model import ResiduosModule
from dataset import ResiduosDataModule


def main():
    torch.set_float32_matmul_precision("high")  # aprovecha Tensor Cores en GPU recientes
    LightningCLI(ResiduosModule, ResiduosDataModule)


if __name__ == "__main__":
    main()
