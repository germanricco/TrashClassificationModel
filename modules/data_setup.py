"""
Contains functionality for creating PyTorch DataLoaders for
image classification data
"""

import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def create_dataloaders(
        train_dir: str,
        test_dir: str,
        transforms: transforms.Compose,
        batch_size: int,
        num_workers: int
):
    """
    Takes in training and testing directories paths and turns
    them into Datasets and then into PyTorch DataLoaders

    Args:
        train_dir:
        test_dir:
        transform:
        batch_size:
        num_workers:

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)

    Example:
        train_dataloader, test_dataloader, class_names = create_dataloaders()
    """

    # Create datasets
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transforms,
                                      target_transform=None)
    test_data = datasets.ImageFolder(root=test_dir,
                                      transform=transforms,
                                      target_transform=None)
    
    class_names = train_data.classes
    

    # Create dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=test_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  shuffle=False)
    
    return train_dataloader, test_dataloader, class_names