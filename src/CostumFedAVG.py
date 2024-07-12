from typing import Tuple, List

import flwr
import numpy as np
from flwr.common import FitIns, Parameters, EvaluateIns
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy


class FedCustomPacketLoss(flwr.server.strategy.FedAvg):

    def __init__(self, fraction_drop: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.fraction_drop = fraction_drop

    def configure_fit(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        rng = np.random.default_rng()
        dropped_clients: int = 0
        valid_clients = []
        for client in clients:
            packet_loss_proba = rng.random()
            if packet_loss_proba < self.fraction_drop:
                dropped_clients += 1
            else:
                valid_clients.append(client)

        if dropped_clients > 0:
            print(str(dropped_clients) + " clients dropped from fitting this round!")
        # Return client/config pairs
        return [(client, fit_ins) for client in valid_clients]

    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        dropped_clients: int = 0
        rng = np.random.default_rng()

        valid_clients = []
        for client in clients:
            packet_loss_proba = rng.random()
            if packet_loss_proba < self.fraction_drop:
                dropped_clients += 1
            else:
                valid_clients.append(client)

        if dropped_clients > 0:
            print(str(dropped_clients) + " clients dropped from evaluation this round!")
        # Return client/config pairs
        return [(client, evaluate_ins) for client in valid_clients]
