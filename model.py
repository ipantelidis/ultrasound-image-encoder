import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class UltrasoundEncoder(nn.Module):
    def __init__(self, latent_dim, seq_len=4):
        super(UltrasoundEncoder, self).__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.encoder = nn.Sequential(*list(resnet.children())[:-2])
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.flatten = nn.Flatten()
        
        resnet_out_features = resnet.fc.in_features 
        self.fc_mu = nn.Linear(resnet_out_features, latent_dim)
        self.fc_logvar = nn.Linear(resnet_out_features, latent_dim)

        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8), 
            nn.ReLU(),
            nn.Unflatten(1, (64, 8, 8)),  
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   
            nn.Sigmoid()
        )

        self.matching_head = nn.Sequential(
            nn.Linear(resnet_out_features * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.ordering_head = nn.Sequential(
            nn.Linear(resnet_out_features * seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len)
        )

    def forward(self, x, task, x_pair=None):
        if task == 'MIR':
            features = self.encoder(x)
            pooled_features = self.global_pool(features) 
            features_flat = self.flatten(pooled_features)
            mu, logvar = self.fc_mu(features_flat), self.fc_logvar(features_flat)
            z = self.reparameterize(mu, logvar)
            reconstructions = self.reconstruction_head(z)
            reconstructions = torch.nn.functional.interpolate(reconstructions, size=(224, 224), mode='bilinear', align_corners=False)
            return reconstructions, mu, logvar

        elif task == 'PM':
            assert x_pair is not None, "x_pair input is required for Patient Matching (PM) task"
            features = self.encoder(x)
            features_flat = self.flatten(self.global_pool(features))
            features_pair = self.encoder(x_pair)
            features_pair_flat = self.flatten(self.global_pool(features_pair))
            combined_features = torch.cat([features_flat, features_pair_flat], dim=1)
            similarity_score = self.matching_head(combined_features)
            return similarity_score

        elif task == 'IO':
            batch_size, seq_len, channels, height, width = x.size()
            x = x.view(batch_size * seq_len, channels, height, width)
            features = self.encoder(x)
            features_flat = self.flatten(self.global_pool(features))
            features_sequence = features_flat.view(batch_size, seq_len * features_flat.size(1))
            order_pred = self.ordering_head(features_sequence)
            return order_pred

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


