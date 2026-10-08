import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, input_size, latent_size):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_size)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_size)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded