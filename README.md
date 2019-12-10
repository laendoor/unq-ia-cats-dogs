# unq-ia-cats-dogs

* [Enunciado](https://drive.google.com/file/d/1CUEzMNLgS-B0j4g0IMioActdmqxIDwER/view)

## Ejecuciones

### Corrida 1

```
--- Configuración inicial ---
Epochs     : 100
Batch Size : 84
Dataset    : 25000
Train      : 70% (17500)
Test       : 20% (5000)
Validation : 10% (2500)
Img size   : 32
Img path   : train

--- Entrenamiento ---
10% Epoch 10: loss entrenamiento=0.0078, loss validación=0.0078, exactitud=61.8800%
20% Epoch 20: loss entrenamiento=0.0061, loss validación=0.0062, exactitud=74.2800%
30% Epoch 30: loss entrenamiento=0.0046, loss validación=0.0072, exactitud=73.5200%
40% Epoch 40: loss entrenamiento=0.0031, loss validación=0.0121, exactitud=67.5800%
50% Epoch 50: loss entrenamiento=0.0020, loss validación=0.0123, exactitud=71.0200%
60% Epoch 60: loss entrenamiento=0.0011, loss validación=0.0177, exactitud=69.4800%
70% Epoch 70: loss entrenamiento=0.0005, loss validación=0.0216, exactitud=71.3800%
80% Epoch 80: loss entrenamiento=0.0002, loss validación=0.0268, exactitud=71.9800%
90% Epoch 90: loss entrenamiento=0.0001, loss validación=0.0281, exactitud=72.1200%
100% Epoch 100: loss entrenamiento=0.0002, loss validación=0.0301, exactitud=71.8200%

--- Resultados ---
Accuracy  : 0.7196
Precision : 0.7201
Recall    : 0.7126
F1        : 0.7163
Matriz de confusión
 [[914 344]
  [357 885]]
```

### Corrida 2

```
python gato.py --epoch=300 --padding=edge --path=train     

--- Configuración inicial ---
Epochs      : 300
Batch Size  : 84
Dataset     : 25000
Dataset path: train/
Train       : 70% (17500)
Test        : 20% (5000)
Validation  : 10% (2500)
Img padding : edge
Img size    : 32x32 px

--- Entrenamiento ---
20% Epoch 60: loss entrenamiento=0.0008, loss validación=0.0185, exactitud=71.7600%
40% Epoch 120: loss entrenamiento=0.0000, loss validación=0.0345, exactitud=73.0600%
60% Epoch 180: loss entrenamiento=0.0000, loss validación=0.0394, exactitud=73.1600%
80% Epoch 240: loss entrenamiento=0.0000, loss validación=0.0413, exactitud=73.3600%
100% Epoch 300: loss entrenamiento=0.0000, loss validación=0.0424, exactitud=73.4000%

--- Resultados ---
Accuracy  : 0.7448
Precision : 0.7532
Recall    : 0.7378
F1        : 0.7454
Matriz de confusión
 [[928 306]
  [332 934]]
```
