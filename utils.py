"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.

    Example usage:
        save_model(model=UNet_model, target_dir="/content/Output_models",
                    model_name="UNet_bs_16_lr_0.01_Epoch_5.pth")
    """
    # Create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    torch.save(obj=model.state_dict(), f=model_save_path)
    #print(f"[INFO] Saving model to: {model_save_path}")

def display_sample(dataloader, labels_map, fig_size, rows, cols):
    """
    Take a random batch in dataloader and plot the images in the batch, along with the corresponding labels.

    Args:
        dataloaders: A dictionary of DataLoaders for the model to be trained on (dataloaders['train']).
        labels_map: A directory for saving encoded value to label names (e.g. {0: "Ants", 1: "Bees"}).
        fig_size: A tuple to determine the figure size in `plt.figure` (e,g. (8,8)).
        rows: A integer for the number of rows in the subplot.
        cols: A integer for the number of columns in the subplot.
        
    Example usage:
        display_sample(dataloaders['train'], labels_map, fig_size=(8,8), rows=3, cols=3)
    """
    images, labels = next(iter(dataloader))
    
    fig = plt.figure(figsize=fig_size)
    for idx in range(cols*rows):
        fig.add_subplot(rows, cols, idx+1, xticks=[], yticks=[])
        plt.title(labels_map[labels[idx].item()])
        plt.imshow(np.transpose(images[idx]))

    plt.show()
