from typing import List, OrderedDict

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
# CIFAR10 Dataset compromised 60.000 (50.000 training, 10.000 test) 32x32 pixel color photographs with 10 classes
# training set has 5000 photographs per class and
# (0: airplane, 1: automobile, 2: bird, 3: car, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)
from torchvision.datasets import CIFAR10

# MNIST Dataset compromised 70.000 28x28 (60.000 training, 10.000 test) handwritten digits.

BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)
"""


# we also need to set an environment variable for deterministic behaviour in all environments that will use the
# Networks here. we need 'CUBLAS_WORKSPACE_CONFIG:16:8' (slower) or
# 'CUBLAS_WORKSPACE_CONFIG:4096:8' (needs 24 Mib GPU Memory)


def get_device() -> DEVICE:
    return DEVICE


class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        # 3 Input channels for RGB and creates 6 feature maps using kernel of size 5x5
        self.conv1 = nn.Conv2d(3, 6, 5)
        # reducing dimensions of convolution layer with pooling
        self.pool = nn.MaxPool2d(2, 2)
        # another convolution layer to create more feature maps
        self.conv2 = nn.Conv2d(6, 16, 5)
        # fully connected layer with 120 neurons and 16 feature maps of size 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 120 Neurons as input and 84 Neurons for the next layer
        self.fc2 = nn.Linear(120, 84)
        # 84 Neurons as input and 10 Neurons for the next layer (final classes)
        self.fc3 = nn.Linear(84, 10)

    # function used to train the neural net using forward pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # transforming previous input to vectors
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict[{k: torch.Tensor(v) for k, v in params_dict}]
    net.load_state_dict(state_dict, strict=True)


def train(net: Net, trainloader: DataLoader, epochs: int) -> None:
    """Train the network on the training set."""
    if torch.cuda.is_available():
        # print("Current used device for training: " + torch.cuda.get_device_name(0))
        pass
    else:
        print("Cuda not available on current device")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


# Evaluate Model and get loss, accuracy using a test set
def test(net, testloader) -> (float, float):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


def load_model():
    return Net().to(DEVICE)


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("..", train=True, download=True, transform=transform)
    testset = CIFAR10("..", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


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
        train_subset_list = list(range(int(i * train_subset_size), int((i + 1) * train_subset_size)))
        train_subset = torch.utils.data.Subset(trainset, train_subset_list)
        trainloaders.append(DataLoader(train_subset, batch_size=BATCH_SIZE))

        val_subset_list = list(range(int(i * val_subset_size), int((i + 1) * val_subset_size)))
        val_subset = torch.utils.data.Subset(testset, val_subset_list)
        valloaders.append(DataLoader(val_subset, batch_size=BATCH_SIZE))
    return trainloaders, valloaders, testloader


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
        train_subset_list = generate_random_integers(num_integers=subset_size, range_end=49999, seed=seed)
        train_subset = torch.utils.data.Subset(trainset, train_subset_list)
        trainloaders.append(DataLoader(train_subset, batch_size=BATCH_SIZE))

        val_subset_list = generate_random_integers(num_integers=int(subset_size/5), range_end=9999, seed=seed)
        val_subset = torch.utils.data.Subset(testset, val_subset_list)
        valloaders.append(DataLoader(val_subset, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


def generate_random_integers(num_integers=10000, range_start=0, range_end=49999, seed=None):
    if seed is not None:
        random.seed(seed)
    return [random.randint(range_start, range_end) for _ in range(num_integers)]


# trainloaders, valloaders, testloader = load_data()

# Only used for testing this
if __name__ == "__main__":
    net = load_model()
    trainloader, testloader, num_examples = load_data()
    train(net, trainloader, 5)
    loss, accuracy = test(net, testloader)
    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
