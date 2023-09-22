import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.file_list = [file for file in os.listdir(data_path) if file.endswith('.png')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_path, file_name)
        image = Image.open(file_path).convert("L")  # Carrega a imagem em escala de cinza (1 canal)
        
        if self.transform:
            image = self.transform(image)
        
        return image
