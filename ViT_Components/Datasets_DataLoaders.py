import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class Datasets_DataLoaders:
    def __init__(self):
        # Define NUM_WORKERS as an instance variable if it's meant to be shared across methods
        self.NUM_WORKERS = os.cpu_count()
        
    def create_dataloaders(self, train_dir: str, test_dir: str, transform: transforms.Compose, batch_size: int=32, num_workers: int = None):
        # Use the class's NUM_WORKERS if num_workers is not specified
        if num_workers is None:
            num_workers = self.NUM_WORKERS

        # Use ImageFolder to create dataset(s)
        train_data = datasets.ImageFolder(train_dir, transform=transform)
        test_data = datasets.ImageFolder(test_dir, transform=transform)

        # Get class names
        class_names = train_data.classes

        # Turn images into data loaders
        train_dataloader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )
        test_dataloader = DataLoader(
            test_data,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_dataloader, test_dataloader, class_names