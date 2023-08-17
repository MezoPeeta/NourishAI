import torch
from torch.utils import data
from torchvision import datasets


def split_dataset(dataset: datasets, split_size: float = 0.2, device: torch.device = "cpu"):
    """Randomly splits a given dataset into two proportions based on split_size and seed.

    Args:
        dataset (torchvision.datasets): A PyTorch Dataset, typically one from torchvision.datasets.
        split_size (float, optional): How much of the dataset should be split? 
            E.g. split_size=0.2 means there will be a 20% split and an 80% split. Defaults to 0.2.
        device (torch.device): The device to split dataset on

    Returns:
        tuple: (random_split_1, random_split_2) where random_split_1 is of size split_size*len(dataset) and 
            random_split_2 is of size (1-split_size)*len(dataset).
    """
    length_1 = int(len(dataset) * split_size)
    length_2 = len(dataset) - length_1

    print(
        f"[INFO] Splitting dataset of length {len(dataset)} "
        f"into splits of size: {length_1} ({int(split_size * 100)}%), "
        f"{length_2} ({int((1 - split_size) * 100)}%)")

    random_split_1, random_split_2 = data.random_split(dataset,
                                                       lengths=[length_1, length_2],
                                                       generator=torch.Generator(device=device)
                                                       )
    return random_split_1, random_split_2
