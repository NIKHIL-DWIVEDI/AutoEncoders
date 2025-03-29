import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

from tqdm import tqdm
import config
import statistics

from dataset import add_noise
## training function
def trainCAE(model,epochs,learning_rate,train_dataloader,val_dataloader):
    """
    Train a Convolutional Autoencoder (CAE) with the given hyperparameters

    Parameters
    ----------
    model : nn.Module
        The model to be trained
    epochs : int
        The number of epochs to train the model
    learning_rate : float
        The learning rate for the optimizer
    train_dataloader : torch.utils.data.DataLoader
        The DataLoader for the training dataset
    val_dataloader : torch.utils.data.DataLoader
        The DataLoader for the validation dataset

    Returns
    -------
    training_loss : list
        The list of training loss at each epoch
    validation_loss : list
        The list of validation loss at each epoch
    """
    print("Training Starts:\n")
    ## define cost function and optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(),lr=learning_rate)

    ## training loop
    training_loss=[]
    validation_loss=[]
    for epoch in range(epochs):
        ll=[]
        acc=[]
        model.train()
        for img,_ in tqdm(train_dataloader):
            img=img.to(config.DEVICE,non_blocking=True)
            output=model(img)
            loss = criterion(output,img) 
            ll.append(loss.item())
            # acc.append()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        training_loss.append(statistics.mean(ll))

        ## validation loop
        model.eval()
        val_losses=[]
        with torch.no_grad():
            for img,_ in val_dataloader:
                img = img.to(config.DEVICE,non_blocking=True)
                output=model(img)
                loss = criterion(output,img)
                val_losses.append(loss.item())
            validation_loss.append(statistics.mean(val_losses))
        print(f"Epoch {epoch+1}: Training Loss = {training_loss[-1]:.4f}, Validation Loss = {validation_loss[-1]:.4f}")    
    return training_loss,validation_loss


def trainDAE(model,epochs,learning_rate,train_dataloader,val_dataloader):
    print("Training Starts:\n")
    ## define cost function and optimiser
    criterion = nn.MSELoss()
    optimiser = optim.Adam(model.parameters(),lr=learning_rate)

    ## training loop
    training_loss=[]
    validation_loss=[]
    for epoch in range(epochs):
        ll=[]
        acc=[]
        model.train()
        for img,_ in tqdm(train_dataloader):
            img=img.to(config.DEVICE,non_blocking=True)
            noisy_img = add_noise(imgs=img,noise_ratio=config.NOISE_RATIO)
            output=model(noisy_img)
            loss = criterion(output,img) ## compare it with the clean image 
            ll.append(loss.item())
            # acc.append()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        training_loss.append(statistics.mean(ll))

        ## validation loop
        model.eval()
        val_losses=[]
        with torch.no_grad():
            for img,_ in val_dataloader:
                img = img.to(config.DEVICE,non_blocking=True)
                noisy_img = add_noise(imgs=img,noise_ratio=config.NOISE_RATIO)
                output=model(noisy_img)
                loss = criterion(output,img)
                val_losses.append(loss.item())
            validation_loss.append(statistics.mean(val_losses))
        print(f"Epoch {epoch+1}: Training Loss = {training_loss[-1]:.4f}, Validation Loss = {validation_loss[-1]:.4f}")  
    return training_loss,validation_loss

def VAEloss(recon,img,mu,logvar):
    # reconstruction loss is combiantion of mse loss and KL divergence
    mse = F.mse_loss(input=recon, target=img)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return mse + kld

def trainVAE(model,epochs,learning_rate,train_dataloader,val_dataloader):
    print("Training of VAEStarts:\n")
    ## define cost function and optimiser
    optimiser = optim.Adam(model.parameters(),lr=learning_rate)

    ## training loop
    training_loss=[]
    validation_loss=[]
    for epoch in range(epochs):
        ll=[]
        acc=[]
        model.train()
        for img,_ in tqdm(train_dataloader):
            img=img.to(config.DEVICE,non_blocking=True)
            recon,mu,logvar=model(img)
            loss = VAEloss(recon,img,mu,logvar)
            ll.append(loss.item())
            # acc.append()
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        training_loss.append(statistics.mean(ll))

        ## validation loop
        model.eval()
        val_losses=[]
        with torch.no_grad():
            for img,_ in val_dataloader:
                img = img.to(config.DEVICE,non_blocking=True)
                recon,mu,logvar=model(img)
                loss = VAEloss(recon,img,mu,logvar)
                val_losses.append(loss.item())
            validation_loss.append(statistics.mean(val_losses))
        print(f"Epoch {epoch+1}: Training Loss = {training_loss[-1]:.4f}, Validation Loss = {validation_loss[-1]:.4f}")  
    return training_loss,validation_loss

def train_model(model_name, model, epochs, learning_rate, train_dataloader,val_dataloader):
    if model_name.lower() == 'cae':
        return trainCAE(model=model, epochs=epochs, learning_rate=learning_rate, train_dataloader=train_dataloader,val_dataloader=val_dataloader)
    if model_name.lower() == 'dae':
        return trainDAE(model=model,epochs=epochs,learning_rate=learning_rate,train_dataloader=train_dataloader,val_dataloader=val_dataloader)
    if model_name.lower() == 'vae':
        return trainVAE(model=model,epochs=epochs,learning_rate=learning_rate,train_dataloader=train_dataloader,val_dataloader=val_dataloader)
    else:
        raise ValueError(f"Training function for model '{model_name}' is not defined.")
