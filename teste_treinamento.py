import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path


import pdb


import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import timm
#assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory


import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler
from util.custom_dataset import CustomDataset
import util.transform_npy as transform_npy


import models_mae

from engine_pretrain import train_one_epoch


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import logging
import os


from torch.nn import DataParallel

from functools import partial



# Verifique se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configurar o diretório de logs e salvamento de modelos
log_dir = 'D:/dados_tcc/output_dir/logs'  # Substitua 'logs' pelo diretório desejado
model_save_path = 'D:/dados_tcc/output_dir/models'  # Substitua 'saved_models' pelo diretório desejado
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_save_path, exist_ok=True)


# Configurar o nível de registro desejado (por exemplo, INFO)
logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)



# Definir hiperparâmetros
batch_size = 16
learning_rate = 0.001
input_size = 512
num_epochs = 10
num_workers = 2


data_path = "D:/dados_tcc"

# Carregar dados
transform_train = transforms.Compose([
        transform_npy.ResizeNpyWithPadding((input_size, input_size)),
        transform_npy.RandomHorizontalFlipNpy(),
        transform_npy.RandomRotationNpy(degrees=(-15, 15)),
        transforms.Lambda(lambda data: data.copy()),  # Copy the data to avoid grad error
        transforms.ToTensor(),
        transforms.Normalize(mean=[130.10511327778235], std=[316.09062860899644])
])
    
# Create a training dataset using the defined transformations
train_dataset = CustomDataset(data_path=os.path.join(data_path, 'train'), transform=transform_train)



# Crie um DataLoader personalizado
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)



# Definir o modelo (incluindo a definição da classe MaskedAutoencoderViT)

# Criar uma instância do modelo e movê-lo para a GPU
model = models_mae.MaskedAutoencoderViT(
        img_size = 512, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer= partial(nn.LayerNorm, eps=1e-6))

model.to(device)
model = DataParallel(model)  # Habilite o treinamento paralelo


# Definir a função de perda e otimizador
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from tqdm import tqdm  # Importe a função tqdm

# Loop de treinamento
for epoch in range(num_epochs):
    print(f'Época [{epoch+1}/{num_epochs}]')

    progress_bar = tqdm(train_loader, desc=f'Época [{epoch+1}/{num_epochs}]', leave=False)
    
    for batch_idx, batch in enumerate(progress_bar):
        images = batch.to(device).float()

        optimizer.zero_grad()
        loss, _, _ = model(images)
        loss.backward()
        optimizer.step()

        progress_bar.set_postfix(loss=loss.item())


    # Registre informações no arquivo de log, se desejar
    logging.info(f'Época [{epoch+1}/{num_epochs}], Perda: {loss.item()}')


# Salvar o modelo treinado
model_save_file = os.path.join(model_save_path, 'mae_vit_model.pth')
torch.save(model.state_dict(), model_save_file)
logging.info(f'Modelo salvo em {model_save_file}')

# Treinamento concluído
logging.info('Treinamento concluído.')
