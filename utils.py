"""Utilidades: callback que registra resultados y costo de cada corrida.

El callback ``RegistroResultados`` se declara en los YAML de configs/. Al terminar
el entrenamiento carga el mejor modelo, lo evalua en validacion y guarda en
``resultados/<backbone>_semilla<semilla>.json``: F1 macro, exactitud, mejor epoca,
historia por epoca, parametros, tamano, minutos de entrenamiento y milisegundos por
foto en CPU. report.py junta esos archivos en la tabla comparativa y las figuras.
"""

import copy
import json
import os
import time

import numpy as np
import torch
import lightning as L
from sklearn.metrics import f1_score, accuracy_score


def contar_parametros(modelo: torch.nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def medir_inferencia_cpu(modelo: torch.nn.Module, img_size: int = 224, hilos: int = 4,
                         repeticiones: int = 30) -> float:
    """Mediana de milisegundos por foto, una foto a la vez, en CPU con ``hilos`` hilos.

    Es una aproximacion a la Raspberry Pi 5 (4 nucleos); no sustituye la medicion
    en el dispositivo.
    """
    m = copy.deepcopy(modelo).to("cpu").eval()
    antes = torch.get_num_threads()
    torch.set_num_threads(hilos)
    x = torch.randn(1, 3, img_size, img_size)
    tiempos = []
    with torch.no_grad():
        for i in range(repeticiones + 5):
            t0 = time.perf_counter()
            m(x)
            if i >= 5:  # las primeras 5 son de calentamiento
                tiempos.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(antes)
    return float(np.median(tiempos))


@torch.no_grad()
def predecir(modelo, cargador, dispositivo):
    modelo.eval().to(dispositivo)
    reales, predichas = [], []
    for x, y in cargador:
        predichas += modelo(x.to(dispositivo)).argmax(1).cpu().tolist()
        reales += y.tolist()
    return reales, predichas


class RegistroResultados(L.Callback):
    """Registra historia, desempeno en validacion y costo de cada corrida."""

    def __init__(self, dir_resultados: str = "resultados", hilos_cpu: int = 4):
        super().__init__()
        self.dir_resultados = dir_resultados
        self.hilos_cpu = hilos_cpu

    def on_fit_start(self, trainer, pl_module):
        self.inicio = time.time()
        self.historia = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        valor = lambda k: float(m[k]) if k in m else None
        self.historia.append({
            "epoca": trainer.current_epoch + 1,
            "perdida_ent": valor("train/loss"),
            "perdida_val": valor("val/loss"),
            "f1_val": valor("val/f1"),
            "exactitud_val": valor("val/acc"),
        })

    def on_fit_end(self, trainer, pl_module):
        minutos = (time.time() - self.inicio) / 60
        ruta_ckpt = trainer.checkpoint_callback.best_model_path
        mejor = type(pl_module).load_from_checkpoint(ruta_ckpt, map_location="cpu") if ruta_ckpt else pl_module

        reales, predichas = predecir(mejor, trainer.datamodule.val_dataloader(), pl_module.device)
        backbone = pl_module.hparams.backbone
        semilla = int(os.environ.get("PL_GLOBAL_SEED", -1))
        nombre = f"{backbone}_semilla{semilla}"

        os.makedirs(os.path.join(self.dir_resultados, "pesos"), exist_ok=True)
        ruta_pesos = os.path.join(self.dir_resultados, "pesos", f"{nombre}.pt")
        torch.save(mejor.net.state_dict(), ruta_pesos)

        f1s = [h["f1_val"] if h["f1_val"] is not None else -1 for h in self.historia]
        resultado = {
            "modelo": backbone,
            "semilla": semilla,
            "version_tensorboard": getattr(trainer.logger, "version", None),
            "checkpoint": ruta_ckpt,
            "mejor_epoca": int(np.argmax(f1s)) + 1 if f1s else None,
            "f1_macro_val": f1_score(reales, predichas, average="macro", zero_division=0),
            "exactitud_val": accuracy_score(reales, predichas),
            "parametros_millones": contar_parametros(mejor.net) / 1e6,
            "tamano_MB": os.path.getsize(ruta_pesos) / 1e6,
            "minutos_entrenamiento": minutos,
            "ms_por_foto_cpu": medir_inferencia_cpu(mejor.net, trainer.datamodule.hparams.img_size,
                                                    self.hilos_cpu),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "hiperparametros": {k: v for k, v in pl_module.hparams.items()},
            "historia": self.historia,
            "reales_val": reales,
            "predichas_val": predichas,
        }
        os.makedirs(self.dir_resultados, exist_ok=True)
        with open(os.path.join(self.dir_resultados, f"{nombre}.json"), "w") as f:
            json.dump(resultado, f, indent=1)
        print(f"\n{nombre}: F1 macro validacion {resultado['f1_macro_val']:.3f} "
              f"(mejor epoca {resultado['mejor_epoca']}), {minutos:.1f} min, "
              f"{resultado['parametros_millones']:.1f} M parametros, "
              f"{resultado['ms_por_foto_cpu']:.0f} ms por foto en CPU")
