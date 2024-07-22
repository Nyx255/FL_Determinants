import time
from collections import OrderedDict
from enum import Enum
from typing import List, Tuple

import flwr as fl
import torch

from flwr.common import Metrics

from src import datasets, cifar10_net, mnist_net
from src.CostumFedAVG import FedCustomPacketLoss

net = None
train_loaders, val_loaders, test_loader = None, None, None
torch.manual_seed(0)

client_latency: float = 0  # in Seconds
client_dropout_rate: float = 0  # in percent


class DatasetEnum(Enum):
    MNIST = 0
    CIFAR10 = 1


current_sim_dataset: DatasetEnum


class FlowerClient(fl.client.NumPyClient):

    def __init__(self, net, train_loader, val_loader, test_loader):
        self.net = net
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def get_parameters(self, config):

        simulate_latency(client_latency)

        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def fit(self, parameters, config):

        simulate_latency(client_latency)

        set_parameters(net, parameters)

        match current_sim_dataset:
            case DatasetEnum.MNIST:
                mnist_net.train(net, self.train_loader, epochs=5)
            case DatasetEnum.CIFAR10:
                cifar10_net.train(net, self.train_loader, epochs=5)

        return self.get_parameters({}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):

        simulate_latency(client_latency)

        set_parameters(net, parameters)

        match current_sim_dataset:
            case DatasetEnum.MNIST:
                loss, accuracy = mnist_net.test(net, self.test_loader)
            case DatasetEnum.CIFAR10:
                loss, accuracy = cifar10_net.test(net, self.test_loader)

        return float(loss), len(self.val_loader.dataset), {"accuracy": accuracy}


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    return model


def client_fn(cid: str):
    """Create a Flower client representing a single organization."""
    # Note: each client gets a different train_loader/val_loader, so each client
    # will train and evaluate on their own unique data
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    # Create a single Flower client representing a single organization
    return FlowerClient(net, train_loader, val_loader, test_loader).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# This Functions starts a Server and n:Clients which then connect to the server for federated learning.
def start_cifar_10_sim(_num_clients: int, _num_rounds: int):
    # Create FedAvg strategy
    strategy = FedCustomPacketLoss(
        fraction_drop=client_dropout_rate,
        fraction_fit=1.0,
        fraction_evaluate=1,
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


# This Functions starts a Server and n:Clients which then connect to the server for federated learning.
def start_mnist_10_sim(_num_clients: int, _num_rounds: int):
    # Create FedAvg strategy
    strategy = FedCustomPacketLoss(
        fraction_drop=client_dropout_rate,
        fraction_fit=1.0,
        fraction_evaluate=1,
        min_fit_clients=_num_clients,
        min_evaluate_clients=int(_num_clients / 2),
        min_available_clients=_num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(mnist_net.get_parameters(mnist_net.Net())),
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
    )

    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if mnist_net.DEVICE.type == "cuda":
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


def prepare_cifar_10_sim(num_clients: int, num_rounds: int, set_size: int, bias_ratio: float = 0.0):
    global net
    net = cifar10_net.load_model()

    train_set, test_set = datasets.download_cifar_10()
    global train_loaders, val_loaders, test_loader

    # cifar train set has size of 50.000
    # cifar test set has size of 10.000
    set_ratio: float = set_size / 50000
    train_loaders, val_loaders, test_loader = datasets.create_loaders_3(train_set, test_set,
                                                                        set_ratio, num_clients, bias_ratio)
    # Start clients, using RAM load distribution. If number of clients is bigger than ram capacity,
    # only load more clients if available
    start_cifar_10_sim(num_clients, num_rounds)


def prepare_mnist_sim(num_clients: int, num_rounds: int, set_size: int, bias_ratio: float = 0.0):
    global net
    net = mnist_net.load_model()

    train_set, test_set = datasets.download_mnist()
    global train_loaders, val_loaders, test_loader

    # mnist train set has size of 60.000
    # mnist test set has size of 10.000
    set_ratio: float = set_size / 60000
    train_loaders, val_loaders, test_loader = datasets.create_loaders_3(train_set, test_set,
                                                                        set_ratio, num_clients, bias_ratio)
    # Start clients, using RAM load distribution. If number of clients is bigger than ram capacity,
    # only load more clients if available
    start_mnist_10_sim(num_clients, num_rounds)


def simulate_latency(latency_ms: float):
    if latency_ms == 0:
        return
    # print("Simulating latency for: " + str(latency_ms / 1000) + " seconds.")
    time.sleep(latency_ms / 1000)


def simulate(sim_dataset: DatasetEnum, clients: int, rounds: int, subset_size: int,
             bias_ratio: float = 0.0, latency: float = 0, dropout_rate: float = 0):
    global current_sim_dataset
    current_sim_dataset = sim_dataset
    # Add Latency to all messages send from the client
    global client_latency
    # client latency in ms
    client_latency = latency

    global client_dropout_rate
    # client latency in ms
    client_dropout_rate = dropout_rate

    match current_sim_dataset:
        case DatasetEnum.MNIST:
            prepare_mnist_sim(clients, rounds, subset_size, bias_ratio)
        case DatasetEnum.CIFAR10:
            prepare_cifar_10_sim(clients, rounds, subset_size, bias_ratio)


if __name__ == '__main__':
    simulate(DatasetEnum.CIFAR10, clients=8, rounds=8, subset_size=1000, bias_ratio=0)
