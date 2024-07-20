import math
from typing import List, OrderedDict, Tuple

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
# CIFAR10 Dataset compromised 60.000 (50.000 training, 10.000 test) 32x32 pixel color photographs with 10 classes
# training set has 5000 photographs per class and
# (0: airplane, 1: automobile, 2: bird, 3: car, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)
from torchvision.datasets import CIFAR10

# MNIST Dataset compromised 70.000 28x28 (60.000 training, 10.000 test) handwritten digits.
from torchvision.datasets import MNIST

SEED: int = 42

BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
np.random.seed(SEED)


def download_mnist() -> Tuple[Dataset, Dataset]:
    """Load MNIST (training and test set)."""
    transform = transforms.Compose(
        # mean value: 0.1307, standard deviation: 0.3081 of MNIST set
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def download_cifar_10() -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("..", train=True, download=True, transform=transform)
    testset = CIFAR10("..", train=False, download=True, transform=transform)
    return trainset, testset


def create_loaders(train_set, test_set, subset_size: int = None, num_splits: int = 1, shuffle: bool = False,
                   biased: bool = False):
    train_loaders = []
    val_loaders = []
    if subset_size is None:
        # splitting train/test-sets into number of clients
        train_subset_size = int(len(train_set) / num_splits)
        val_subset_size = int(len(test_set) / num_splits)
    else:
        # splitting train/test-sets into number of clients with even sized data
        train_subset_size = subset_size
        val_subset_size = int(subset_size * len(test_set) / len(train_set))

    for i in range(num_splits):
        if biased:
            train_subset_loader = create_biased_loader(train_set, train_subset_size, bias_ratio=0.5, seed=SEED)
        elif shuffle:
            train_subset_loader = create_random_loader(train_set, train_subset_size, int(i * train_subset_size),
                                                       int((i + 1) * train_subset_size), SEED)
        else:
            train_subset_loader = create_loader(train_set, int(i * train_subset_size), int((i + 1) * train_subset_size))
        train_loaders.append(train_subset_loader)

        if biased:
            val_subset_loader = create_biased_loader(train_set, val_subset_size, seed=SEED)
        elif shuffle:
            val_subset_loader = create_random_loader(test_set, val_subset_size, int(i * val_subset_size),
                                                     int((i + 1) * val_subset_size), SEED)
        else:
            val_subset_loader = create_loader(test_set, int(i * val_subset_size), int((i + 1) * val_subset_size))

        val_loaders.append(val_subset_loader)

    test_loader = create_loader(test_set)
    return train_loaders, val_loaders, test_loader


def create_loaders_2(train_set, test_set, subset_size: int = None, num_splits: int = 1, shuffle: bool = False,
                     biased: bool = False):
    train_loaders = []
    val_loaders = []
    if subset_size is None:
        # splitting train/test-sets into number of clients
        train_subset_size = int(len(train_set) / num_splits)
        val_subset_size = int(len(test_set) / num_splits)
    else:
        # splitting train/test-sets into number of clients with even sized data
        train_subset_size = subset_size
        val_subset_size = int(subset_size * len(test_set) / len(train_set))

    for i in range(num_splits):
        if biased:
            train_subset_loader = create_biased_loader(train_set, train_subset_size, bias_ratio=0.5, seed=SEED)
        elif shuffle:
            train_subset_loader = create_random_loader(train_set, train_subset_size, int(i * train_subset_size),
                                                       int((i + 1) * train_subset_size), SEED)
        else:
            train_subset_loader = create_loader(train_set, int(i * train_subset_size), int((i + 1) * train_subset_size))
        train_loaders.append(train_subset_loader)

        if biased:
            val_subset_loader = create_biased_loader(train_set, val_subset_size, seed=SEED)
        elif shuffle:
            val_subset_loader = create_random_loader(test_set, val_subset_size, int(i * val_subset_size),
                                                     int((i + 1) * val_subset_size), SEED)
        else:
            val_subset_loader = create_loader(test_set, int(i * val_subset_size), int((i + 1) * val_subset_size))

        val_loaders.append(val_subset_loader)

    test_loader = create_loader(test_set)
    return train_loaders, val_loaders, test_loader


def create_loader(dataset, start: int = 0, end: int = None):
    if end is None:
        end = len(dataset)
    sub_set_list = list(range(start, end))
    sub_set = Subset(dataset, sub_set_list)
    return DataLoader(sub_set, batch_size=BATCH_SIZE)


def create_subsets(train_set, test_set, set_ration: float = 0.1, subset_count: int = 1,
                   bias_ratio: float = 0.0, shuffle: bool = False):
    # Guards
    if set_ration < 0.0 or set_ration > 1.0:
        print("Set Ratio cant be bigger then 1.0 or smaller then 0.0! Exiting...")
        raise SystemExit

    if bias_ratio < 0.0 or bias_ratio > 1.0:
        print("Bias Ratio cant be bigger then 1.0 or smaller then 0.0! Exiting...")
        raise SystemExit

    train_subset_size: int = int(math.floor(len(train_set) * set_ration))
    print("Create train subset Size of: " + str(train_subset_size) + " set of Size: " + str(len(train_set)))
    if train_subset_size * subset_count > 1:
        print("Subset count is too big for Dataset with current subset ratio! ")
        raise SystemExit

    test_subset_size: int = int(math.floor(len(test_set) * set_ration))
    print("Create test subset Size of: " + str(test_subset_size) + " set of Size: " + str(len(test_set)))
    if test_subset_size * subset_count > 1:
        print("Subset count is too big for Dataset with current subset ratio! ")
        raise SystemExit
    # Guards end

    classes: list = list(set(train_set.targets))
    train_subsets: list[Subset] = []
    test_subsets: list[Subset] = []

    for i in range(0, subset_count):
        # select random class as bias for train and test set
        random_class = random.choice(classes)

        train_subset = create_biased_subset(train_set, train_subset_size, random_class, bias_ratio)
        train_subsets.append(train_subset)

        test_subset = create_biased_subset(test_set, test_subset_size, random_class, bias_ratio)
        test_subsets.append(test_subset)

    return train_subsets, test_subsets


def create_biased_subset(set, subset_size: int, class_bias: int, bias_ratio: float = 0.5):
    # split dataset.targets between matching classes and other classes
    class_indices = [i for i, target in enumerate(set.targets) if target == class_bias]
    other_indices = [i for i, target in enumerate(set.targets) if target != class_bias]

    # shuffle lists
    random.shuffle(class_indices)
    random.shuffle(other_indices)

    # choose amount of biased classes and fill rest with random classes
    num_biased_samples = int(subset_size * bias_ratio)
    num_other_samples = subset_size - num_biased_samples

    selected_indices = class_indices[:num_biased_samples] + other_indices[:num_other_samples]
    biased_subset = Subset(set, selected_indices)
    return biased_subset


def create_random_loader(dataset, set_size: int, range_start: int = 0, range_end: int = None, seed: int = None):
    if range_end is None:
        sub_set_list = generate_random_integers(num_integers=set_size, range_start=range_start,
                                                range_end=len(dataset), seed=seed)
    else:
        sub_set_list = generate_random_integers(num_integers=set_size, range_start=range_start,
                                                range_end=range_end, seed=seed)
    sub_set = Subset(dataset, sub_set_list)
    return DataLoader(sub_set, batch_size=BATCH_SIZE)


def create_biased_loader(dataset, set_size: int, bias_ratio: float = 0.5, seed: int = None):
    """
    Generates a DataLoader with a specified size, biased towards a random class.
    :param dataset: PyTorch dataset
    :param set_size: size of the new dataset
    :param bias_ratio: ratio of samples that should belong to the biased class
    :param seed: seed used for randomization
    :return: DataLoader
    """
    if seed is not None:
        random.seed(seed)
    random_class = random.choice(list(set(dataset.targets)))
    print("Selected bias class: " + str(random_class))
    # split dataset.targets between matching classes and other classes
    class_indices = [i for i, target in enumerate(dataset.targets) if target == random_class]
    other_indices = [i for i, target in enumerate(dataset.targets) if target != random_class]

    # shuffle lists
    random.shuffle(class_indices)
    random.shuffle(other_indices)

    # choose amount of biased classes and fill rest with random classes
    num_biased_samples = int(set_size * bias_ratio)
    num_other_samples = set_size - num_biased_samples

    selected_indices = class_indices[:num_biased_samples] + other_indices[:num_other_samples]
    biased_subset = Subset(dataset, selected_indices)

    return DataLoader(biased_subset, batch_size=len(biased_subset), shuffle=True)


# Example usage (assuming you have a dataset object)
# loader = create_biased_loader(my_dataset, set_size=100, bias_ratio=0.5, seed=42)


def generate_random_integers(num_integers: int, range_start: int, range_end: int, seed=None):
    """
    Generates random list of integers.
    :param num_integers: size of list we will generate
    :param range_start: start value for range
    :param range_end: end value for range
    :param seed:
    :return: List of random integers in range
    """
    if seed is not None:
        random.seed(seed)
    return [random.randint(range_start, range_end) for _ in range(num_integers)]
