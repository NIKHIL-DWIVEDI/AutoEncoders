import config
from dataset import dataset_details,get_dataloader,get_dataset,train_val_split
from models.convolutionalAE import CAE
import argparse
from models.init import get_model

from utils import plot_train_val_loss
from train import train_model

def parse_args():
    parser = argparse.ArgumentParser(description="Different types of AutoEncoders")
    parser.add_argument('model',type=str,choices=['cae','vae','dae','sparse'],help='Type of autoencoder (cae: Convolutional, vae: Variational, dae: Denoising, sparse: Sparse)')
    return parser.parse_args()

if __name__=="__main__":
    args = parse_args()
    ## download the dataset
    datasets = get_dataset('cifar10',train=True,download=True) 
    ## split the dataset into train and val
    train_dataset,val_dataset = train_val_split(datasets,config.TRAIN_RATIO)
    ##cerate the dataloader
    train_dataloader = get_dataloader(dataset=train_dataset,batch_size=config.BATCH_SIZE,shuffle=config.SHUFFLE)
    val_dataloader = get_dataloader(dataset=val_dataset,batch_size=config.BATCH_SIZE,shuffle=False)
    ## details about the dataset and visualize it
    # dataset_details(dataset=datasets,dataloader=dataloader)

    ## create the instance of the model
    model = get_model(args.model)
    model=model.to(config.DEVICE)

    ## define training loop
    training_loss,validation_loss = train_model(args.model,model,epochs=config.EPOCHS,learning_rate=config.LEARNING_RATE,train_dataloader=train_dataloader,val_dataloader=val_dataloader)

    ## plot the training loss
    plot_train_val_loss(args.model,training_loss=training_loss,validation_loss=validation_loss)



