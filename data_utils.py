from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SubsetRandomSampler
from torchvision import transforms, datasets


def load_client_data(client_idx):
    # Load the data for a specific client
    raise NotImplementedError
    return None


class ListDataset(Dataset):
    def __init__(self, data: List[torch.Tuple[torch.Tensor, torch.Tensor]]):
        self.data = data
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)


def filter_dataset(dataset: Dataset, classes: List[int], n_samples: int):
    # Filter dataset for specific classes and n samples per class
    filtered_data = []
    class_counts = {c: 0 for c in classes}
    for data, target in dataset:
        if target in classes and class_counts[target] < n_samples:
            filtered_data.append((data, target))
            class_counts[target] += 1
    return ListDataset(filtered_data)


def load_dataset(classes=None, samples_per_class=None):
    """
    Load FashionMNIST dataset
    :param classes: list of classes to filter
    :param n_samples: number of samples per class
    :return: train_loader, val_loader, len(train_loader)
    """

    # Check if class and n_samples are both None or both not None
    if (classes is not None and samples_per_class is None) or (samples_per_class is not None and classes is None):
        raise ValueError("n_samples should be specified if classes is specified")
                                                       
    # Define transforms
    trf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    trainset = datasets.FashionMNIST('./dataset', train=True, download=True, transform=trf)
    valset = datasets.FashionMNIST('./dataset', train=False, transform=trf)

    # Filter dataset for specific classes and n samples per class
    if classes is not None and samples_per_class is not None:
        trainset = filter_dataset(trainset, classes, samples_per_class)
        valset = filter_dataset(valset, classes, samples_per_class)
    return trainset, valset


def load_dataloader(trainset, valset, client_id, num_samples, batch_size=64, nworkers=4):
    # Load the data for a specific client
    start_index = client_id*num_samples
    sampler = SubsetRandomSampler(range(start_index, start_index + num_samples))

    # Create dataloaders
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=nworkers, sampler=sampler)
    val_loader = DataLoader(valset, batch_size=batch_size, num_workers=nworkers, sampler=sampler)
    return train_loader, val_loader
