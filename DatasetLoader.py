import torch
from torchvision import datasets


def load_dataset(dataset_path, transform, batch_size, num_workers, shuffle):

    # load training and validation set
    train_set = dataset_path + '\\train'
    valid_set = dataset_path + '\\valid'

    train_data = datasets.ImageFolder(train_set, transform=transform)
    valid_data = datasets.ImageFolder(valid_set, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_data,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    return train_loader, valid_loader
