import torch
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

# Custom dataset.
class ImageDataset(Dataset):
    def __init__(self, images, labels=None, tfms=None):
        self.X = images
        self.y = labels

        # Apply Augmentations if training.
        if tfms == 0: # If validating.
            self.aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
            ])
        else: # If training.
            self.aug = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                transforms.RandomAutocontrast(p=0.5),
                transforms.RandomGrayscale(p=0.5),
                transforms.RandomRotation(45),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    )
            ])
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        image = Image.open(self.X[i])
        image = image.convert('RGB')
        image = self.aug(image)
        label = self.y[i].astype(np.int32)
        return (
            image,
            torch.tensor(label, dtype=torch.long)
        )


def get_datasets():
    # Read the data.csv file and get the image paths and labels.
    df = pd.read_csv('task2/inputs/trainval.csv')
    X = df.image_path.values # Image paths.
    y = df.target.values # Targets
    (xtrain, xtest, ytrain, ytest) = train_test_split(
        X, y,
        test_size=0.20, random_state=42
    )
    dataset_train = ImageDataset(xtrain, ytrain, tfms=1)
    dataset_valid = ImageDataset(xtest, ytest, tfms=0)
    return dataset_train, dataset_valid

def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers=0):
    """
    Prepares the training and validation data loaders.

    :param dataset_train: The training dataset.
    :param dataset_valid: The validation dataset.

    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers
    )
    valid_loader = DataLoader(
        dataset_valid, batch_size=batch_size, 
        shuffle=False, num_workers=num_workers
    )
    return train_loader, valid_loader 