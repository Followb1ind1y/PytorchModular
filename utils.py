"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch

from pathlib import Path


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    """
    Saves a PyTorch model to a target directory.

    Args:
        model: A target PyTorch model to save.
        target_dir: A directory for saving the model to.
        model_name: A filename for the saved model. Should include either ".pth" or ".pt" as the file extension.

    Example:
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