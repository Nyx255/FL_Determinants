from typing import List, OrderedDict, Tuple

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
# CIFAR10 Dataset compromised 60.000 (50.000 training, 10.000 test) 32x32 pixel color photographs with 10 classes
# training set has 5000 photographs per class and
# (0: airplane, 1: automobile, 2: bird, 3: car, 4: deer, 5: dog, 6: frog, 7: horse, 8: ship, 9: truck)
from torchvision.datasets import CIFAR10

# MNIST Dataset compromised 70.000 28x28 (60.000 training, 10.000 test) handwritten digits.
from torchvision.datasets import MNIST

from src import cifar10_net

BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)


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


# Only used for testing this
if __name__ == "__main__":
    net = load_model()
    train_set, test_set = datasets.download_cifar_10()
    train_loaders, val_loaders = datasets.create_loaders(train_set, test_set)
    epochs = 5

    train(net, train_loaders[0], epochs)
    loss, accuracy = test(net, val_loaders[0])

    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
