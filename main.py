import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import UltrasoundEncoder
from dataset import MultiTaskDataset
from training import train
from utils import loss_fn_mir, loss_fn_pm, loss_fn_io
from config import data_path, checkpoint_path, batch_size, lr_encoder, lr_default, num_epochs

# data loading
df = pd.read_csv(data_path)

# device 
device = 'cuda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485], std=[0.229]),  
])

# datasets
dataset_mir = MultiTaskDataset(df, task_type='MIR', transform=transform)
dataset_pm = MultiTaskDataset(df, task_type='PM', transform=transform)
dataset_io = MultiTaskDataset(df, task_type='IO', transform=transform)

# dataloaders 
loader_mir = DataLoader(dataset_mir, batch_size=batch_size, shuffle=True)
loader_pm = DataLoader(dataset_pm, batch_size=batch_size, shuffle=True)
loader_io = DataLoader(dataset_io, batch_size=batch_size, shuffle=True)

# model
ultrasound_encoder = UltrasoundEncoder(latent_dim=64, seq_len=4).to(device)

# optimizer 
optimizer = optim.Adam([
    {'params': ultrasound_encoder.encoder.parameters(), 'lr': lr_encoder},
    {'params': ultrasound_encoder.reconstruction_head.parameters()},
    {'params': ultrasound_encoder.matching_head.parameters()},
    {'params': ultrasound_encoder.ordering_head.parameters()}
], lr=lr_default)


def main():
    train(
        ultrasound_encoder, 
        {
            'mir': loader_mir, 
            'pm': loader_pm, 
            'io': loader_io
        }, 
        optimizer, 
        num_epochs=num_epochs, 
        loss_fn_mir=loss_fn_mir, 
        loss_fn_pm=loss_fn_pm, 
        loss_fn_io=loss_fn_io, 
        device=device,
        checkpoint_path=checkpoint_path
    )

if __name__ == '__main__':
    main()
