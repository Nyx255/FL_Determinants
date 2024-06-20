from typing import List, OrderedDict, Tuple

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# CIFAR10 Dataset compromised 60.000 (50.000 training, 10.000 test) 32x32 pixel color photographs with 10 classes
# training set has 5000 photographs per class and
# (0: airplane, 1: automobile, 2: bird, 3: car, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)
from torchvision.datasets import CIFAR10

# MNIST Dataset compromised 70.000 28x28 (60.000 training, 10.000 test) handwritten digits.
from torchvision.datasets import MNIST

BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)


def _download_mnist() -> Tuple[Dataset, Dataset]:
    """Downloads (if necessary) and returns the MNIST dataset.

    Returns
    -------
    Tuple[MNIST, MNIST]
        The dataset for training and the dataset for testing MNIST.
    """
    transform = transforms.Compose(
        # mean value: 0.1307, standard deviation: 0.3081 of MNIST set
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)
    return trainset, testset


def _download_cifar_10() -> Tuple[Dataset, Dataset]:
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("..", train=True, download=True, transform=transform)
    testset = CIFAR10("..", train=False, download=True, transform=transform)
    return trainset, testset


def load_data(train_set, test_set):
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)
    return train_loader, test_loader


def test(train_set, test_set, subset_size: int = None, num_splits: int = 1):
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
        pass


def load_datasets(num_clients: int, subset_size: int = -1):
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("..", train=True, download=True, transform=transform)
    testset = CIFAR10("..", train=False, download=True, transform=transform)

    trainloaders = []
    valloaders = []

    if subset_size != -1:
        if num_clients * subset_size > 50000:
            print("num clients: " + str(num_clients) + ", subset size: " + str(subset_size) + ", result: " + str(
                num_clients * subset_size))
            raise IndexError("(Subset Size * Number of Clients) is too big for train-set splitting")
        elif subset_size / 5 > 10000:
            raise IndexError("Subset Size too big for test-set")

    """
    if len(trainset) % num_clients != 0 or len(testset) % num_clients != 0:
        print("Train or Test set not divisible by num clients! This will omit non divisible training data")
    """
    if subset_size == -1:
        # splitting train/test-sets into number of clients
        train_subset_size = int(len(trainset) / num_clients)
        val_subset_size = int(len(testset) / num_clients)
        testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    else:
        # splitting train/test-sets into number of clients with even sized data
        train_subset_size = subset_size
        val_subset_size = int(subset_size * len(testset) / len(trainset))

        test_subset_list = list(range(0, subset_size))
        test_subset = torch.utils.data.Subset(trainset, test_subset_list)
        testloader = DataLoader(test_subset, batch_size=BATCH_SIZE)
    print("train-subset size: " + str(train_subset_size))
    print("validation-subset size: " + str(val_subset_size))
    for i in range(num_clients):
        train_subset_loader = create_loader(trainset, int(i * train_subset_size), int((i + 1) * train_subset_size))
        trainloaders.append(train_subset_loader)

        val_subset_loader = create_loader(testset, int(i * val_subset_size), int((i + 1) * val_subset_size))
        valloaders.append(val_subset_loader)
    return trainloaders, valloaders, testloader


def create_loader(dataset, start: int = 0, end: int = None):
    if end is None:
        end = len(dataset)
    sub_set_list = list(range(start, end))
    sub_set = torch.utils.data.Subset(dataset, sub_set_list)
    return DataLoader(sub_set, batch_size=BATCH_SIZE)


def create_random_loader(dataset, set_size: int, range_start: int = 0, range_end: int = None, seed: int = None):
    if range_end is None:
        sub_set_list = generate_random_integers(num_integers=set_size, range_start=range_start,
                                                range_end=len(dataset), seed=seed)
    else:
        sub_set_list = generate_random_integers(num_integers=set_size, range_start=range_start,
                                                range_end=range_end, seed=seed)
    sub_set = torch.utils.data.Subset(dataset, sub_set_list)
    return DataLoader(sub_set, batch_size=BATCH_SIZE)


def load_randomized_dataset(num_clients: int, subset_size: int, seed=None):
    if subset_size > 50000:
        print("subset size is too big! " + str(subset_size) + " > 50.000 !")
        return
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("..", train=True, download=True, transform=transform)
    testset = CIFAR10("..", train=False, download=True, transform=transform)
    test_subset_size = subset_size / 5

    trainloaders = []
    valloaders = []

    for i in range(num_clients):
        train_subset_list = generate_random_integers(num_integers=subset_size, range_end=49999,
                                                     seed=seed + (num_clients * 100) + i)
        train_subset = torch.utils.data.Subset(trainset, train_subset_list)
        trainloaders.append(DataLoader(train_subset, batch_size=BATCH_SIZE))

        val_subset_list = generate_random_integers(num_integers=int(subset_size / 5), range_end=9999,
                                                   seed=seed + (num_clients * 100) + i)
        val_subset = torch.utils.data.Subset(testset, val_subset_list)
        valloaders.append(DataLoader(val_subset, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


def generate_random_integers(num_integers: int, range_start: int, range_end: int, seed=None):
    if seed is not None:
        random.seed(seed)
    return [random.randint(range_start, range_end) for _ in range(num_integers)]
