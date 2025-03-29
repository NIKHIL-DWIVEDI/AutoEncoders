import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
import torch
import config
from config import NOISE_RATIO
import matplotlib.pyplot as plt
from torch.utils.data import random_split


def get_transforms():
    return transforms.Compose([
        transforms.ToTensor() ## convert the PIL image to tensor
    ])

def get_dataset(dataset_name='cifar10', train=True, download=True):
    transform = get_transforms()
    
    if dataset_name.lower() == 'cifar10':
        dataset = datasets.CIFAR10(
            root='./data/',
            train=train,
            transform=transform,
            download=download
        )
    return dataset

def get_dataloader(dataset, batch_size=config.BATCH_SIZE, shuffle=config.SHUFFLE):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )

def dataset_details(dataset,dataloader):
    ## explore the dataset
    print("The total no. of images: ",len(dataset)) # total no. of images = 50000
    print("Shape of the image: ",dataset[0][0].shape) # so dataset contains the tuple(image,label) # shape of the image is [3,32,32]

    print("Sample image shape after dataloader:", next(iter(dataloader))[0].shape) # [batch_size,channel,height,width]

    ## visualise the first image and corresponding label
    for img,label in iter(dataloader):
        print(img.shape,label) ## the first batch
        plt.imshow(img[0].permute(1,2,0))
        plt.show()
        print("the corresponding label to the image: ",label[0])
        break

def add_noise(imgs,noise_ratio=NOISE_RATIO):
    noisy_imgs = imgs + noise_ratio*torch.randn_like(imgs)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)  # Keep pixels in [0,1]
    return noisy_imgs

def train_val_split(dataset,train_ratio):
    train_size = int (train_ratio*len(dataset))
    val_size = len(dataset)-train_size
    return random_split(dataset=dataset,lengths=[train_size,val_size])
