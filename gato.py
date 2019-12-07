import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataset
from catnet import CatNet
from dataset import CatDogDataset
from train_test import train, test, print_data, dataloader
from sklearn.metrics import confusion_matrix, accuracy_score

batch_size = 84             # cantidad de archivos entran por batch de entrenamiento
test_proportion = .2        # proporción de archivos a usar de test (ej: 20%)
validation_proportion = .1  # proporción de archivos a usar de test (ej: 10%)
img_path = 'train/min'      # path de las imágenes FIXME: uso min para que no sea tan pesado, cambiar a train/

# Datasets
catdog_dataset = CatDogDataset(data_dir=img_path)
len_dataset = len(catdog_dataset)

test_size = int(test_proportion * len_dataset)
validation_size = int(validation_proportion * len_dataset)
train_size = len_dataset - test_size - validation_size

train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
    catdog_dataset, [train_size, test_size, validation_size])

# Loaders
train_loader = dataloader(train_dataset, batch_size)
test_loader = dataloader(test_dataset, batch_size)
validation_loader = dataloader(validation_dataset, batch_size)

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


# Ponemos el modelo en modo evaluación
model.eval()

# Predicciones para los datos de validación
# Primero generamos la matriz de entradas y vector de resultados a partir del dataloader
validation_inputs = list()
validation_real = list()
for batch, tensor in enumerate(validation_loader):
    value, output = tensor
    validation_inputs.append(value)
    validation_real.append(output)
inputs = torch.cat(validation_inputs)                 # Se pasan a formato Tensor
validation_real = torch.cat(validation_real)
_, validation_predicted = torch.max(model(inputs), 1)  # Se obtienen las predicciones


# Armamos la matriz de confusión
cm = confusion_matrix(validation_real.numpy(), validation_predicted.numpy())

# Prints para chequear data
print("real:      ", validation_real.numpy())
print("predicted: ", validation_predicted.numpy())
print(cm)

# Graficamos la matriz de confusión
tick_marks, labels = [0, 1], ['Gato', 'Perro']
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
plt.xticks(tick_marks, labels, rotation=45)
plt.yticks(tick_marks, labels)
plt.xlabel("El modelo predijo que era")
plt.ylabel("La imagen real era")
plt.show()

# Evaluamos Accuracy
accuracy = accuracy_score(validation_real, validation_predicted.numpy())
print(f'EPOCHS: {epochs} - ACCURACY: {accuracy}')


# TODO: se debe proveer: Accuracy, Precision, Recall y F1
