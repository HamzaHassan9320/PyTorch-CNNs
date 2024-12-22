import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_data_loaders(batch_size=128, num_workers=2):
    """
    Download and create train/test DataLoaders for CIFAR-10"
    """

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616))
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
        (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = datasets.CIFAR10(
        root='data',
        train=True,
        transform=train_transforms,
        download=True
    )

    test_dataset = datasets.CIFAR10(
        root='data',
        train=False,
        transform=test_transforms,
        download=True
    )

    train_loader = DataLoader(train_dataset,
                             batch_size= batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                             drop_last=True)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             drop_last=True
    )

    return train_loader, test_loader