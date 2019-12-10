import csv
import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataset
import sklearn.metrics as skm
from catnet import CatNet
from dataset import CatDogDataset
from train_test import train, test, print_data, dataloader


epochs = 300
dataset_path = 'test'
img_size = 32
padding_mode = 'edge'
batch_size = 84
FILENAME_MODEL = 'gato_300.pt'
catdog_dataset = CatDogDataset(data_dir=dataset_path, img_size=img_size, padding_mode=padding_mode, label_mode='eval')

validation_size = len(catdog_dataset)
# validation_dataset = torch.utils.data.random_split(catdog_dataset, [validation_size])
validation_loader = dataloader(catdog_dataset, batch_size)

# Loading model
model = CatNet()
model.load_state_dict(torch.load(FILENAME_MODEL))
model.eval()  # modo evaluación

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
validation_predicted = torch.softmax(model(inputs), 1)[1]   # Se obtienen las predicciones

print("validation real: ", validation_real)
# print("validation inputs: ", validation_predicted)

# Genero el archivo con los resultados
with open('submission.csv', mode='w') as file:
    writer = csv.writer(file)
    writer.writerow(['id', 'label'])
    for idx, p in enumerate(validation_predicted.detach().numpy()):
        print(validation_real.numpy()[idx], "{:.4f}".format(p))
        writer.writerow([validation_real.numpy()[idx], "{:.4f}".format(p)])
