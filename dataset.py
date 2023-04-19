"""
Contains example datasets for Object Detection and Semantic Segmentation.
"""
import os
import re
import numpy as np
import torch

from torchvision import transforms
from PIL import Image

class ObjectDetectionDataset(torch.utils.data.Dataset):
    """
    Object Detection Dataset Example.

    Args:
        root: The root directory contains the image folder and annotation folder.
        transforms: Optional transform to be applied on a sample.

    Example Usage:
        ObjectDetectionDataset(root=os.path.join(output_dir, x), transforms=data_augmentation)
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.img_dir = os.path.join(root, 'PNGImages')
        self.ann_dir  = os.path.join(root, 'Annotation')
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.anns = list(sorted(os.listdir(self.ann_dir)))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        curr_img_dir = os.path.join(self.img_dir, self.imgs[idx])
        curr_ann_dir = os.path.join(self.ann_dir, self.anns[idx])

        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        boxes, labels = self.get_ann_from_txt(curr_ann_dir)

        boxes = torch.FloatTensor(boxes)
        labels = torch.LongTensor(labels)

        if self.transforms:
            transformed = self.transforms(image=image, bboxes=boxes, category_ids=labels)

        tenn = transforms.ToTensor()
        image = tenn(image)

        return image, boxes, labels

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function 
        (to be passed to the DataLoader).

        Args:
            batch: an iterable of N sets from __getitem__()
        
        Returns:
            A tuple contains all the zipped information from batch
        
        Example Usage:
            torch.utils.data.DataLoader(image_datasets[x], batch_size=Batch_Size,collate_fn=image_datasets[x].collate_fn) 
        """
        return tuple(zip(*batch))
    
    def get_ann_from_txt(self, ann_dir):
        """
        The annotation information are stored in the .txt files. We need a function to extract the useful data.

        Args:
            ann_dir: The annotation directory contains .txt files.
        
        Returns:
            A list of bounding box coordinates (e.g. [[xmin, ymin, xmax, ymax], ...]),
            and a list of labels corresponding to these boxes
        """
        annotations, labels = [], []
        with open(ann_dir) as iostream:
            content = iostream.read()

        for line in content.split("\n"):
            if "(Xmin, Ymin) - (Xmax, Ymax)" in line:
                ann_list = [eval(i) for i in re.findall('\d+', line)][1:]
                annotations.append(ann_list)
                labels.append(1)
            
        return annotations, labels

class SemanticSegmentationDataset(torch.utils.data.Dataset):
    """
    Semantic Segmentation Dataset Example.

    Args:
        root: The root directory contains the image folder and mask folder.
        transforms: Optional transform to be applied on a sample.

    Example Usage:
        ObjectDetectionDataset(root=os.path.join(output_dir, x), transforms=data_augmentation)
    """
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        
        self.img_dir = os.path.join(root, 'PNGImages')
        self.mask_dir  = os.path.join(root, 'PedMasks')
        self.imgs = list(sorted(os.listdir(self.img_dir)))
        self.masks = list(sorted(os.listdir(self.mask_dir)))
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        curr_img_dir = os.path.join(self.img_dir, self.imgs[idx])
        curr_mask_dir = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        mask = Image.open(curr_mask_dir)
        mask = np.array(mask)
        mask[mask != 0] = 1

        if self.transforms:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        tenn = transforms.ToTensor()
        image = tenn(image)
        mask = tenn(mask)
        
        return image, mask
    
    def collate_fn(self, batch):
        """
        Since each image may have a different size, we need a collate function 
        (to be passed to the DataLoader).

        Args:
            batch: an iterable of N sets from __getitem__()
        
        Returns:
            A tuple contains all the zipped information from batch
        
        Example Usage:
            torch.utils.data.DataLoader(image_datasets[x], batch_size=Batch_Size,collate_fn=image_datasets[x].collate_fn) 
        """
        return tuple(zip(*batch))