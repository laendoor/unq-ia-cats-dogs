import os
import random
import numpy as np
from PIL import Image
from scanf import scanf
import torch.utils.data.dataset
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

dir_images = 'train/sample'
cant_archivos = 0


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size=0):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir, x) for x in files]
        if data_size < 0 or data_size > len(files):
            assert "Data size should be between 0 to number of files in the dataset"
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        width, height = image.size

        padding_pair = (height - width, 0) if width < height else (0, width - height)
        image = transforms.Pad(padding_pair, padding_mode='edge')(image)
        image = transforms.Resize((128, 128))(image)

        image = np.array(image)
        image = image / 255
        image = image.transpose(2, 0, 1)
        image = torch.tensor(image)
        return image, self.label(image_address)

    @staticmethod
    def label(filename):
        _, _, label, _ = scanf("%s/%s/%s.%d.jpg", filename)
        return 0 if label == "cat" else 1  # Kaggle convention


def mostrarImagen(dataset, nroImagen):
    imagen, etiqueta = dataset[nroImagen]
    imagen = imagen.numpy()
    imagen = imagen.transpose(1, 2, 0)
    print(etiqueta)
    plt.imshow(imagen)
    plt.title(etiqueta)
    plt.show()


catdog_dataset = CatDogDataset(data_dir=dir_images, data_size=cant_archivos)

mostrarImagen(catdog_dataset, 2)
