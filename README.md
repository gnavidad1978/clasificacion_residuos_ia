# Clasificación de residuos por material: redes convolucionales contra Vision Transformer

Proyecto final de la materia Inteligencia Artificial (DCI019), Doctorado en Ciencias e Ingeniería, Universidad Autónoma de Guadalajara. Autora: Gabriela Medina Navidad.

El proyecto compara tres redes convolucionales (MobileNetV3, ResNet-50 y YOLO11n-cls) contra un Vision Transformer (ViT-B/16) en la clasificación de residuos en seis categorías derivadas de la NAE-SEMADES-007/2008 de Jalisco: vidrio, plástico, metal, papel y cartón, residuos inorgánicos de difícil reciclaje y orgánico. Se reportan desempeño y costo (parámetros, tamaño, tiempo de entrenamiento y tiempo por foto), porque el modelo elegido debe funcionar en un sistema embebido. Es un avance de la tesis doctoral sobre visión artificial embebida para la validación de residuos antes de su depósito.

La organización sigue la del repositorio de la clase [clase_arquitectura_cugdl](https://github.com/UlisesMoyaSanchez/clase_arquitectura_cugdl): PyTorch Lightning con LightningCLI y archivos YAML, de modo que los parámetros se cambian sin tocar el código.

## Estructura

```
clasificacion_residuos_ia/
├── main.py              punto de entrada (LightningCLI)
├── model.py             fábrica de redes y LightningModule
├── dataset.py           LightningDataModule que lee la partición fija
├── prepare_data.py      descarga RealWaste, acomodo a seis categorías, partición, exploración y línea base
├── utils.py             callback que guarda desempeño y costo de cada corrida
├── benchmark.py         velocidad de cada red en GPU con datos sintéticos (costo de cómputo)
├── report.py            tabla comparativa, F1 por categoría, curvas, matrices de confusión y errores
├── configs/             un YAML por modelo
│   ├── mobilenet_v3.yaml
│   ├── resnet50.yaml
│   ├── yolo11n_cls.yaml
│   └── vit_b_16.yaml
├── notebooks/
│   └── Proyecto_final.ipynb   cuaderno de Colab que ejecuta y presenta el proyecto
└── requirements.txt
```

Los datos, los registros de TensorBoard, los modelos y los resultados no se suben al repositorio (ver `.gitignore`); en Colab se guardan en Google Drive.

## Cómo ejecutar el cuaderno en Colab

1. Abrir el cuaderno directamente desde GitHub con este enlace: https://colab.research.google.com/github/gnavidad1978/clasificacion_residuos_ia/blob/main/notebooks/Proyecto_final.ipynb
2. Activar la GPU: Entorno de ejecución, Cambiar tipo de entorno de ejecución, GPU T4, Guardar.
3. Ejecutar todo: Entorno de ejecución, Ejecutar todo (Ctrl+F9).
4. Aceptar los dos avisos: que el cuaderno viene de GitHub (Ejecutar de todos modos) y el permiso para conectar Google Drive.
5. El archivo de RealWaste debe estar en `MyDrive/proyecto_IA/realwaste.zip`; si no está, `prepare_data.py` lo descarga (unos 657 MB).

Tiempos aproximados con GPU T4: alrededor de 8 minutos por entrenamiento, es decir cerca de hora y media para los 4 modelos con 3 semillas. Para una prueba rápida se puede dejar `SEMILLAS = [42]` en la primera celda.

Si Colab se desconecta, no se pierde lo terminado: cada entrenamiento guarda su resultado en `MyDrive/proyecto_IA/repo_resultados` y, al volver a ejecutar todo, la celda de entrenamiento salta lo que ya existe. Conviene evitar que la computadora se suspenda mientras entrena (cargador conectado y suspensión en Nunca).

Mientras entrena, si TensorBoard muestra *Data could not be loaded*, no afecta el entrenamiento; basta con volver a ejecutar esa celda al terminar.

El conjunto de prueba se evalúa una sola vez, al final, quitando el comentario de la última celda del cuaderno.

En resumen, el cuaderno ejecuta:

```bash
git clone https://github.com/gnavidad1978/clasificacion_residuos_ia.git
cd clasificacion_residuos_ia
pip install -r requirements.txt
python prepare_data.py --zip /content/drive/MyDrive/proyecto_IA/realwaste.zip
python main.py fit --config configs/mobilenet_v3.yaml --seed_everything 42
python benchmark.py
python report.py
```

## Cambiar parámetros sin tocar el código

Editando el YAML, o agregando el parámetro al comando:

```bash
# Otra semilla
python main.py fit --config configs/vit_b_16.yaml --seed_everything 7

# Más épocas y otra tasa de aprendizaje
python main.py fit --config configs/resnet50.yaml --trainer.max_epochs 20 --model.lr 0.0001

# SGD con momentum en lugar de AdamW
python main.py fit --config configs/mobilenet_v3.yaml --model.optimizer sgd --model.lr 0.01

# Prueba rápida en CPU
python main.py fit --config configs/mobilenet_v3.yaml --trainer.precision 32-true --trainer.max_epochs 1

# Ver todos los parámetros disponibles
python main.py fit --help
```

Si se cambia `--model.backbone` desde la línea de comandos, conviene cambiar también `--trainer.logger.init_args.name` para que TensorBoard la guarde con su nombre.

## Protocolo experimental

1. **Semilla fija.** `seed_everything` en cada YAML; `prepare_data.py` usa `np.random.default_rng(semilla)`. La partición es la misma en todas las corridas; las semillas de entrenamiento (42, 7 y 2026) cambian el orden de las fotos, los aumentos y la inicialización de la última capa.
2. **Partición antes de mirar los datos.** 70 % entrenamiento, 15 % validación y 15 % prueba, estratificada por categoría. Se reparten grupos de fotos casi idénticas (huella dHash de 16 por 16, umbral de 10 bits), no fotos sueltas, para que una foto y su casi gemela no queden en conjuntos distintos. La exploración y las figuras usan solo entrenamiento.
3. **Línea base.** Clase mayoritaria de entrenamiento, evaluada en validación (`resultados/linea_base.json`).
4. **Métrica elegida de antemano.** F1 macro, porque las categorías no están balanceadas y todas pesan igual en el bote. Con ella se elige la mejor época (ModelCheckpoint) y se detiene el entrenamiento (EarlyStopping, paciencia de 4). Se reportan además exactitud y matriz de confusión.

La normalización usa los valores publicados de ImageNet, que no se calculan con nuestros datos y por tanto no filtran información de validación ni de prueba. Los aumentos (volteo horizontal, giro de hasta 15 grados, brillo, contraste y saturación) solo se aplican a entrenamiento; no se usa volteo vertical porque un residuo sostenido en la mano no aparece de cabeza. El conjunto de prueba se usa una sola vez, al final, con `python main.py test`.

YOLO11n-cls se entrena con el mismo bucle, aumentos y regla de paro que las demás redes, no con el entrenador de Ultralytics, para que la única diferencia entre modelos sea la arquitectura.

El tiempo por foto se mide en CPU con 4 hilos y una foto a la vez, como aproximación a una Raspberry Pi 5; no sustituye la medición en el dispositivo. El tiempo total de entrenamiento no sirve para comparar arquitecturas en Colab gratuito, porque leer y transformar las fotos con 2 procesadores tarda lo mismo para cualquier red y cada modelo se detiene en una época distinta; por eso `benchmark.py` mide las fotos por segundo de cada red en GPU con datos sintéticos, y el reporte muestra segundos por época.

## Datos

RealWaste: S. Single, S. Iranmanesh y R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning," *Information*, vol. 14, no. 12, p. 633, 2023, doi: 10.3390/info14120633. Conjunto en UCI Machine Learning Repository, doi: 10.24432/C5SS4G, licencia CC BY 4.0.

Acomodo a las seis categorías: Cardboard y Paper a papel y cartón; Glass a vidrio; Metal a metal; Plastic a plástico; Food Organics y Vegetation a orgánico; Miscellaneous Trash a difícil reciclaje (interpretación propia, en revisión por muestreo; se puede excluir con `--excluir-misc`); Textile Trash se excluye porque el campus no tiene bote de textiles.

Limitaciones declaradas: RealWaste no contiene residuos sostenidos en la mano, se tomó en Australia y no publica identificadores de objeto, por lo que fotos del mismo objeto desde ángulos distintos pueden quedar en conjuntos diferentes. Algunas etiquetas originales no coinciden con la NAE-SEMADES-007/2008 (por ejemplo papel aluminio, envolturas metalizadas y envases multicapa, que la norma coloca en difícil reciclaje).
