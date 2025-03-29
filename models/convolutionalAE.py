
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

## define convolutional autoencoder
class CAE(nn.Module):
    def __init__(self):
        super().__init__()

        ## encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )

        ## decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=3,stride=1,padding=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x
    