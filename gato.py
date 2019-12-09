import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataset
import sklearn.metrics as skm
from catnet import CatNet
from dataset import CatDogDataset
from train_test import train, test, print_data, dataloader


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', default=100, type=int, metavar='EPOCH', help='Cantidad de Epochs')
parser.add_argument('--padding', default='edge', type=str, metavar='PADDING', help='Cantidad de Epochs')
parser.add_argument('--path', default='train', type=str, metavar='PATH', help='Ruta del dataset')
args = parser.parse_args()

# Configuración
epochs = args.epoch              # cantidad de épocas (iteraciones)
batch_size = 84                  # cantidad de archivos entran por batch de entrenamiento
test_proportion = .2             # proporción de archivos a usar de test (ej: 20%)
validation_proportion = .1       # proporción de archivos a usar de test (ej: 10%)
img_size = 32                    # tamaño de resize para aplicarle al dataset (ej: 32x32 px)
padding_mode = args.padding      # tipo de padding para generar imágenes cuadradas
dataset_path = args.path         # path de las imágenes

# Datasets
catdog_dataset = CatDogDataset(data_dir=dataset_path, img_size=img_size, padding_mode=padding_mode)
len_dataset = len(catdog_dataset)

test_size = int(test_proportion * len_dataset)
validation_size = int(validation_proportion * len_dataset)
train_size = len_dataset - test_size - validation_size

train_dataset, test_dataset, validation_dataset = torch.utils.data.random_split(
    catdog_dataset, [train_size, test_size, validation_size])

print("--- Configuración inicial ---")
print('Epochs      : {:d}'.format(epochs))
print('Batch Size  : {:d}'.format(batch_size))
print('Dataset     : {:d}'.format(len_dataset))
print('Dataset path: {:s}/'.format(dataset_path))
print('Train       : {:.0f}% ({:d})'.format((1 - test_proportion - validation_proportion) * 100, train_size))
print('Test        : {:.0f}% ({:d})'.format(test_proportion * 100, test_size))
print('Validation  : {:.0f}% ({:d})'.format(validation_proportion * 100, validation_size))
print('Img padding : {:s}'.format(padding_mode))
print('Img size    : {:d}x{:d} px'.format(img_size, img_size))


# Loaders
train_loader = dataloader(train_dataset, batch_size)
test_loader = dataloader(test_dataset, batch_size)
validation_loader = dataloader(validation_dataset, batch_size)

# Creamos el modelo
model = CatNet()
loss_criteria = nn.CrossEntropyLoss()  # criterio de loss: CrossEntropyLoss está pensado para clasificación

# Optimizer: Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

# Entrenamiento
epoch_nums = []
training_loss = []
validation_loss = []
show_every = round(epochs * .2)  # show epoch info every xx% of processing
print("\n--- Entrenamiento ---")
for epoch in range(1, epochs + 1):
    train_loss = train(model, train_loader, optimizer, loss_criteria)  # train
    test_loss, accuracy = test(model, test_loader, loss_criteria)  # test

    # Guardamos los datos en las listas
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

    # Cada 10 iteraciones vamos imprimiendo nuestros resultados parciales
    if epoch % show_every == 0:
        print_data(round(epoch / epochs * 100), epoch, train_loss, test_loss, accuracy)

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
inputs = torch.cat(validation_inputs)  # Se pasan a formato Tensor
validation_real = torch.cat(validation_real)
_, validation_predicted = torch.max(model(inputs), 1)  # Se obtienen las predicciones

# Armamos la matriz de confusión
cm = skm.confusion_matrix(validation_real.numpy(), validation_predicted.numpy())

# Graficamos la matriz de confusión
# tick_marks, labels = [0, 1], ['Gato', 'Perro']
# plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
# plt.colorbar()
# plt.xticks(tick_marks, labels, rotation=45)
# plt.yticks(tick_marks, labels)
# plt.xlabel("El modelo predijo que era")
# plt.ylabel("La imagen real era")
# plt.show()


# Evaluamos
accuracy = skm.accuracy_score(validation_real.numpy(), validation_predicted.numpy())
precision = skm.precision_score(validation_real.numpy(), validation_predicted.numpy())
recall = skm.recall_score(validation_real.numpy(), validation_predicted.numpy())
f1 = skm.f1_score(validation_real.numpy(), validation_predicted.numpy())

# Resultados
print("\n--- Resultados ---")
print('Accuracy  : {:.4f}'.format(accuracy))
print('Precision : {:.4f}'.format(precision))
print('Recall    : {:.4f}'.format(recall))
print('F1        : {:.4f}'.format(f1))
print('Matriz de confusión\n', cm)
