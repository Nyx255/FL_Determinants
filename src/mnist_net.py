from typing import List, OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src import datasets

BATCH_SIZE = 128
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)


def get_device() -> DEVICE:
    return DEVICE


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        output_tensor = F.relu(self.conv1(input_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = F.relu(self.conv2(output_tensor))
        output_tensor = self.pool(output_tensor)
        output_tensor = nn.Flatten()(output_tensor)
        output_tensor = F.relu(self.fc1(output_tensor))
        output_tensor = self.fc2(output_tensor)
        return output_tensor


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict[{k: torch.Tensor(v) for k, v in params_dict}]
    net.load_state_dict(state_dict, strict=True)


def train(net: nn.Module, trainloader: DataLoader, epochs: int) -> nn.Module:
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    if torch.cuda.is_available():
        # print("Current used device for training: " + torch.cuda.get_device_name(0))
        pass
    else:
        print("Cuda not available on current device")
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
    return net


# Evaluate Model and get loss, accuracy using a test set
def test(net: nn.Module, testloader: DataLoader) -> (float, float):
    """Validate the network on the entire test set."""
    if len(testloader.dataset) == 0:
        raise ValueError("Testloader size is zero!")
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def load_model():
    return Net().to(DEVICE)


# Only used for testing this
if __name__ == "__main__":
    net = load_model()
    train_set, test_set = datasets.download_mnist()
    train_loaders, val_loaders, test_loader = datasets.create_loaders(train_set, test_set)
    epochs = 5

    train(net, train_loaders[0], epochs)
    loss, accuracy = test(net, val_loaders[0])

    print(f"Loss: {loss:.5f}, Accuracy: {accuracy:.3f}")
