import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataset
from catnet import CatNet
from dataset import CatDogDataset
from train_test import train, test, print_data
from sklearn.metrics import confusion_matrix

batch_size = 84             # cantidad de archivos entran por batch de entrenamiento
test_proportion = .2        # proporción de archivos a usar de test (ej: 20%)
validation_proportion = .1  # proporción de archivos a usar de test (ej: 10%)
img_path = 'train/min'      # path de las imágenes FIXME: uso min para que no sea tan pesado, cambiar a train/

# Creamos los Datasets y los loaders para el entrenamiento
catdog_dataset = CatDogDataset(data_dir=img_path)
len_dataset = len(catdog_dataset)

test_size = int(test_proportion * len_dataset)
validation_size = int(validation_proportion * len_dataset)
train_size = len_dataset - test_size - validation_size

train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
    catdog_dataset, [train_size, test_size, validation_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# Creamos el modelo
model = CatNet()

input('Ya estoy listo. Enter para entrenar...')

loss_criteria = nn.CrossEntropyLoss()  # criterio de loss: CrossEntropyLoss está pensado para clasificación


# optimizer: Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

# listas vacías para ir guardando loss y validación en cada iteración
epoch_nums = []
training_loss = []
validation_loss = []


# Entrenamiento
epochs = 100
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, loss_criteria)   # train
    test_loss, accuracy = test(model, test_loader, loss_criteria)       # test

    # Guardamos los datos en las listas
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

    # Cada 10 iteraciones vamos imprimiendo nuestros resultados parciales
    if epoch % 10 == 0:
        print_data(epoch, train_loss, test_loss, accuracy)


# Creamos la matriz de confusión, esta es parte del paquete scikit
# Ponemos el modelo en modo evaluación
model.eval()

# Hacemos las predicciones para los datos de test
# Para eso, en primer lugar generamos la matriz de entradas y vector de
# resultados a partir del dataloader
entradas = list()
salidas = list()
for batch, tensor in enumerate(test_loader):
    valor, salida = tensor
    entradas.append(valor)
    salidas.append(salida)
# Se pasan a formato Tensor
entradas = torch.cat(entradas)
salidas = torch.cat(salidas)
# Se obtienen las predicciones
_, predicted = torch.max(model(entradas), 1)


# Graficamos la matriz de confusión
tick_marks = [0, 1]
labels = ['Gato', 'Perro']
cm = confusion_matrix(salidas.numpy(), predicted.numpy())
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("El modelo predijo que era")
plt.ylabel("La imagen real era")
plt.show()


# TODO: se debe proveer: Accuracy, Precision, Recall y F1
