import argparse
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.utils.data.dataset
import sklearn.metrics as skm
from catnet import CatNet
from dataset import CatDogDataset
from train_test import train, test, print_data, dataloader

# def predict_image(image):
#     image_tensor = test_transforms(image).float()
#     image_tensor = image_tensor.unsqueeze_(0)
#     input = Variable(image_tensor)
#     input = input.to(device)
#     output = model(input)
#     index = output.data.cpu().numpy().argmax()
#     return index


epochs = 300
dataset_path = 'test_10'
img_size = 32
padding_mode = 'edge'
batch_size = 84
FILENAME_MODEL = 'gato_{:d}.pt'.format(epochs)
catdog_dataset = CatDogDataset(data_dir=dataset_path, img_size=img_size, padding_mode=padding_mode, label_mode='eval')

validation_size = len(catdog_dataset)
validation_dataset = torch.utils.data.random_split(catdog_dataset, [validation_size])
validation_loader = dataloader(validation_dataset[0], batch_size)

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
# _, validation_predicted = torch.max(model(inputs), 1)  # Se obtienen las predicciones
validation_predicted = torch.sigmoid(torch.mean(model(inputs), 1))   # Se obtienen las predicciones

print("validation real: ", validation_real)
print("validation inputs: ", validation_predicted)
#
# import csv
#
# with open('submission.csv', mode='w') as file:
#     writer = csv.writer(file, delimiter='')
#     writer.writerow(['id', 'valor'])

for p in validation_predicted.detach().numpy():
    print("{:.2f}".format(p))
