from dataset import CatDogDataset
import matplotlib.pyplot as plt

dir_images = 'train/sample'


def mostrarImagen(dataset, nroImagen):
    imagen, etiqueta = dataset[nroImagen]
    imagen = imagen.numpy()
    imagen = imagen.transpose(1, 2, 0)
    print(etiqueta)
    plt.imshow(imagen)
    plt.title(etiqueta)
    plt.show()


catdog_dataset = CatDogDataset(data_dir=dir_images)

mostrarImagen(catdog_dataset, 2)
