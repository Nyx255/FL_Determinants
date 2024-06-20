from collections import OrderedDict
from typing import List, Tuple

import flwr as fl
import torch

from flwr.common import Metrics

from src import datasets, cifar10_net
from src.cifar10_net import load_model, train, test

net = load_model()
train_loaders, val_loaders, test_loader = None, None, None
torch.manual_seed(0)


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net, train_loader, val_loader):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):
        set_parameters(net, parameters)
        train(net, self.train_loader, epochs=5)
        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        set_parameters(net, parameters)
        loss, accuracy = test(net, self.val_loader)
        return float(loss), len(self.val_loader.dataset), {"accuracy": accuracy}


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


# Obsolete?
"""
def start_client(train_loader, test_loader) -> None:
    client = FlowerClient(train_loader, test_loader).to_client()
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=client,
    )


def start_clients(num_clients: int) -> None:
    train_loaders, val_loaders, test_loader = load_datasets(num_clients)
    for i in range(num_clients):
        start_client(train_loaders[i], test_loader)
"""


def client_fn(cid: str):
    """Create a Flower client representing a single organization."""
    # Load data (CIFAR-10)
    # Note: each client gets a different train_loader/val_loader, so each client
    # will train and evaluate on their own unique data
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    # Create a single Flower client representing a single organization
    return FlowerClient(net, train_loader, val_loader).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# This Functions starts a Server and n:Clients which then connect to the server for federated learning.
def start_multiple_client_simulation(_num_clients: int, _num_rounds: int):
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=_num_clients,
        min_evaluate_clients=int(_num_clients / 2),
        min_available_clients=_num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(cifar10_net.get_parameters(cifar10_net.Net())),
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if cifar10_net.DEVICE.type == "cuda":
        # here we are assigning an entire GPU for each client.
        print("Using GPU for clients.")
        client_resources = {"num_cpus": 2, "num_gpus": 0.2}
        # Refer to our documentation for more details about Flower Simulations
        # and how to set up these `client_resources`.

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=_num_clients,
        config=fl.server.ServerConfig(num_rounds=_num_rounds),
        strategy=strategy,
        client_resources=client_resources,
    )


def start_cifar_10_sim(num_clients: int, num_rounds: int, subset_size: int):
    train_set, test_set = datasets.download_cifar_10()
    global train_loaders, val_loaders, test_loader
    train_loaders, val_loaders, test_loader = datasets.create_loaders(train_set, test_set,
                                                                      subset_size, num_splits=num_clients)
    # Start clients, using RAM load distribution. If number of clients is bigger than ram capacity,
    # only load more clients if available
    start_multiple_client_simulation(num_clients, num_rounds)


def start_mnist_sim(num_clients: int, num_rounds: int, subset_size: int):
    train_set, test_set = datasets.download_mnist()
    global train_loaders, val_loaders, test_loader
    train_loaders, val_loaders, test_loader = datasets.create_loaders(train_set, test_set,
                                                                      subset_size, num_splits=num_clients)
    # Start clients, using RAM load distribution. If number of clients is bigger than ram capacity,
    # only load more clients if available
    start_multiple_client_simulation(num_clients, num_rounds)


if __name__ == '__main__':
    clients: int = 4
    rounds: int = 4
    set_per_client: int = 10000
    start_cifar_10_sim(clients, rounds, set_per_client)
