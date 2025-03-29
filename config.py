import torch
import os

## check for gpu
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")  # Use MPS on Apple Silicon
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")  # Use CUDA if an NVIDIA GPU is available
else:
    DEVICE = torch.device("cpu")  # Default to CPU

print("Using DEVICE:", DEVICE)


# Set random seeds for reproducibility
OUTPUT_DIR = "./results"
DATASET_DIR = './data'
SAVE_PATH = './results/plots/'


os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SAVE_PATH, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_WORKERS = 2
PIN_MEMORY = True
SHUFFLE = True
NOISE_RATIO=0.3
TRAIN_RATIO=0.8
