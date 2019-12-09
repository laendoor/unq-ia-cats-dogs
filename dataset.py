import os
import random
import numpy as np
from PIL import Image
from scanf import scanf
import torch.utils.data.dataset
import torchvision.transforms as transforms


class CatDogDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, data_size=0, img_size=32, padding_mode='edge'):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir, x) for x in files if '.gitkeep' not in x]

        if data_size < 0 or data_size > len(files):
            assert "Data size should be between 0 to number of files in the dataset"

        self.img_size = img_size
        self.padding_mode = padding_mode
        self.data_size = len(files) if data_size == 0 else data_size
        self.files = random.sample(files, self.data_size)

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        width, height = image.size

        padding_pair = (height - width, 0) if width < height else (0, width - height)
        image = transforms.Pad(padding_pair, padding_mode=self.padding_mode)(image)
        image = transforms.Resize((self.img_size, self.img_size))(image)

        image = np.array(image)
        image = image / 255
        image = image.transpose(2, 0, 1)
        image = torch.Tensor(image)
        return image, self.label(image_address)

    @staticmethod
    def label(filename):
        _, label, _ = scanf("%s/%s.%d.jpg", filename)
        return 0 if label == "cat" else 1  # Kaggle convention
