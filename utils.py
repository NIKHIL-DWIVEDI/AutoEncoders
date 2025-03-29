import torch
import os
import matplotlib.pyplot as plt
from config import OUTPUT_DIR
from datetime import datetime
from config import SAVE_PATH

import matplotlib.pyplot as plt
from datetime import datetime
import os
from config import SAVE_PATH

def plot_train_val_loss(model_name, training_loss, validation_loss):
    plt.plot(range(len(training_loss)), training_loss, label='Training Loss')
    if validation_loss:
        plt.plot(range(len(validation_loss)), validation_loss, label='Validation Loss')
    plt.title(f'Training and Validation Loss for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    if SAVE_PATH:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folderpath = os.path.join(SAVE_PATH, model_name)
        os.makedirs(folderpath, exist_ok=True)
        filepath = os.path.join(folderpath, f"loss_plot_{model_name}_{timestamp}.png")
        plt.savefig(filepath)
    plt.show()


