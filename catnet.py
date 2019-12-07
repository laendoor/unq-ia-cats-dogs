import torch.utils.data.dataset
import torch.nn as nn


# FIXME: revisar definiciones
class CatNet(nn.Module):
    def __init__(self):
        super(CatNet, self).__init__()

        # Capa convolución 1: Toma imágenes con 3 canales y un kernel de
        # 5x5 para generar imágenes de 32 canales
        # Entrada: 32x32; Salida: 28x28 (Se pierden 2 pixels de cada borde)
        self.conv1 = nn.Conv2d(3, 32, 5)

        # Capa MaxPool. Se deja un elemento de cada kernel de 2x2
        # Entrada: 28x28; Salida: 14x14
        self.pool = nn.MaxPool2d(2, 2)

        # Capa convolución 2: Toma imágenes de 32 canales y un kernel de
        # 5x5 para generar imágenes de 16 canales.
        # Entrada: 14x14; Salida: 10x10
        self.conv2 = nn.Conv2d(32, 16, 5)

        # Luego de conv2 se hace otro MaxPool, así que
        # Entrada: 10x10; Salida: 5x5
        # Finalmente llegamos a la red lineal, como teníamos 16 canales
        # y una imagen de 5x5, la cantidad de entradas de la capa es
        # 16x5x5.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
