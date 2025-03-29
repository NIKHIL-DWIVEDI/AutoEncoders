import torch.nn as nn
import torch

class VAE(nn.Module):
    def __init__(self,latent_dim=128):
        super().__init__()

        self.latent_dim=latent_dim
        ## encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=1,padding=1),
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        )
        self.flatten_dim = 64*34*34
        ## after encoder flatten it
        self.fc_mu = nn.Linear(in_features=self.flatten_dim,out_features=latent_dim)
        self.fc_logvar = nn.Linear(in_features=self.flatten_dim,out_features=latent_dim)
        self.fc_decoder = nn.Linear(in_features=latent_dim,out_features=self.flatten_dim)


        ## decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=1, stride=1, padding=1),
            nn.Sigmoid()
        )


    def reparameterize(self,mu,logvar):
        std=torch.exp(0.5*logvar)
        eps=torch.randn_like(std)
        return mu+eps*std

    def forward(self,x):
        x=self.encoder(x) ## encode the input imge
        x=x.view(x.shape[0],-1) ## flatten
        mu=self.fc_mu(x)
        logvar=self.fc_logvar(x)
        z=self.reparameterize(mu,logvar)
        x=self.fc_decoder(z)
        x=x.view(-1,64,34,34)
        recon = self.decoder(x)
        return recon,mu,logvar
        


