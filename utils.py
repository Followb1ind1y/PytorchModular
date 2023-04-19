"""
Contains various utility functions for PyTorch model training and saving.
"""
import os
import cv2
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
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

def display_images(**images):
    """
    Helper function for image visualization. Taking any number of images and plot them in the same row.

    Example Usage:
        display_images(Example1=image2, Example2=Image2)
    """
    num_images = len(images)
    plt.figure(figsize=(15,15))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, num_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=15)
        plt.imshow(image)
    plt.show()

def display_images_labels(dataloader, labels_map, fig_size, rows, cols):
    """
    Take a random batch in dataloader and plot the images in the batch, along with the corresponding labels.

    Args:
        dataloaders: A dictionary of DataLoaders for the model to be trained on (dataloaders['train']).
        labels_map: A directory for saving encoded value to label names (e.g. {0: "Ants", 1: "Bees"}).
        fig_size: A tuple to determine the figure size in `plt.figure` (e,g. (8,8)).
        rows: A integer for the number of rows in the subplot.
        cols: A integer for the number of columns in the subplot.
        
    Example usage:
        display_images_labels(dataloaders['train'], labels_map, fig_size=(8,8), rows=3, cols=3)
    """
    images, labels = next(iter(dataloader))
    
    fig = plt.figure(figsize=fig_size)
    for idx in range(cols*rows):
        fig.add_subplot(rows, cols, idx+1, xticks=[], yticks=[])
        plt.title(labels_map[labels[idx].item()])
        plt.imshow(np.transpose(images[idx]))

    plt.show()

def display_images_boundary(image, boxes, labels, name_dic, color_dic, score = None):
    """
    Taking bounding box information and display the boundary on the original image.
    
    Args:
        image: A image tensor used for plot boundaries.
        boxes: A list tensor contains bounding box coordinates.
        labels: A list tensor contains labels corresponding to boundary boxes.
        name_dic: A dictionary matchs the labels to class names (e.g. {1: 'Person', 2: 'Car'})
        color_dic: A dictionary matchs the labels to boundary boxes' color (e.g. {1: 'palegreen', 2: 'red})
        score: Optional list used to display the prediction scores next to the boundary boxes.
        
    Example Usage:
        display_images_boundary(images[0], boxes[0], labels[0], {1: 'Person'}, {1: 'palegreen'})
    """
    transform = torchvision.transforms.ToPILImage()
    image = transform(image)
    boxes = boxes.tolist()
    labels = labels.tolist()

    img_bbox = ImageDraw.Draw(image)
    new_font = ImageFont.truetype(os.path.join(cv2.__path__[0],'qt','fonts','DejaVuSansCondensed-Bold.ttf'), 10)

    for idx in range(len(boxes)):
        img_bbox.rectangle(boxes[idx], outline=color_dic[labels[idx]], width=2)
        if score == None: 
            img_bbox.text((boxes[idx][0], boxes[idx][1]-15), name_dic[labels[idx]], 
                          font=new_font, align ="left", fill=color_dic[labels[idx]]) 
        else:
            img_bbox.text((boxes[idx][0], boxes[idx][1]-15), name_dic[labels[idx]]+' '+ f"{score[idx].item():.2%}", 
                          font=new_font, align ="left", fill=color_dic[labels[idx]])
    
    return image

def display_images_masks(dataloader, fig_size, rows):
    """
    Take a random batch in dataloader and plot the images in the batch, along with the corresponding labels.

    Args:
        dataloaders: A dictionary of DataLoaders for the model to be trained on (dataloaders['train']).
        fig_size: A tuple to determine the figure size in `plt.figure` (e,g. (8,8)).
        rows: A integer for the number of rows in the subplot.
        
    Example usage:
        display_images_masks(dataloaders['test'],fig_size=(25,4), rows=8)
    """
    images, masks = next(iter(dataloader))
    
    fig = plt.figure(figsize=fig_size)
    for idx in range(rows):
        fig.add_subplot(2, 8, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1,2,0)))
        fig.add_subplot(2, 8, 8+idx+1, xticks=[], yticks=[])
        reverse_encoded = np.transpose(masks[idx],(1,2,0))
        plt.imshow(reverse_encoded)

    plt.show()