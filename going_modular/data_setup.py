"""
contains functionality for creating pytorch dataloaders and for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def create_dataloaders(train_directory: str, test_directory: str, transform: transforms.Compose, batch_size: int):
    """
    Creat training and test dataloader.
    
    Takes in training directory and testing directory path and turns them into PyTorch Datasets and then into PyTorch DataLoader.
    :param train_directory: Path to training directory.
    :param test_directory: Path to testing directory.
    :param transform: Torchvision transforms to perform on training and testing data
    :param batch_size: Number of samples per batch in each of the DataLoaders.
    :return: A tuple of (train_dataloader, test_dataloader, class_names).
            class names is a list of the target classes.
            
    Example usage:
        train_dataloader, test_dataloader, classe_names = creat_dataloader(train_directory=path/to/train_directory, test_directory=path/to/test_directory, transform=some_transform, batch_size=32
    """
    train_data = datasets.ImageFolder(root=train_directory, transform=transform)
    test_data = datasets.ImageFolder(root=test_directory, transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_dataloader, test_dataloader, class_names
