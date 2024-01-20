import itertools
import logging
import random
import time
from typing import Callable, Dict, Optional, Tuple
import flwr as fl
import subprocess
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar
import concurrent.futures
import numpy as np

import torch

from model_utils import visualize_federated_learning_performance, save_experiment_data

# Import Intel extension for pytorch for Intel GPUs
try:
    import intel_extension_for_pytorch as ipex
except ImportError:
    pass

def is_xpu_available() -> bool:
    return hasattr(torch, "xpu") and torch.xpu.is_available()

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif is_xpu_available():
        return "xpu"
    return "cpu"




class FashionTrainingStrategy(fl.server.strategy.FedAvg):
    
    def __init__(
        self, 
        *, 
        fraction_fit: float = 1, 
        fraction_evaluate: float = 1, 
        min_fit_clients: int = 2, 
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2, 
        evaluate_fn: Callable[[int, NDArrays, Dict[str, Scalar]], Tuple[float, Dict[str, Scalar]] | None] | None = None, 
        on_fit_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, 
        on_evaluate_config_fn: Callable[[int], Dict[str, Scalar]] | None = None, 
        accept_failures: bool = True, initial_parameters: Parameters | None = None, 
        fit_metrics_aggregation_fn: MetricsAggregationFn | None = None, 
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None,
    ) -> None:
        if evaluate_metrics_aggregation_fn is None:
            evaluate_metrics_aggregation_fn = self.evaluate_metrics_aggregation_fn
        
        super().__init__(
            fraction_fit=fraction_fit, 
            fraction_evaluate=fraction_evaluate, 
            min_fit_clients=min_fit_clients, 
            min_evaluate_clients=min_evaluate_clients, 
            min_available_clients=min_available_clients, 
            evaluate_fn=evaluate_fn, 
            on_fit_config_fn=on_fit_config_fn, 
            on_evaluate_config_fn=on_evaluate_config_fn, 
            accept_failures=accept_failures, 
            initial_parameters=initial_parameters, 
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn, 
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn
        )

    def evaluate_metrics_aggregation_fn(self, evaluate_res):
        """
        Aggregate the evaluation metrics.
        :param evaluate_res: List of evaluation results from clients. Each element is a tuple (loss, accuracy, ...).
        :return: Aggregated evaluation result.
        """

        metric_list = {}
        for _, metrics in evaluate_res:
            for key, value in metrics.items():
                if key not in metric_list:
                    metric_list[key] = []
                metric_list[key].append(value)
        
        # Compute mean loss and accuracy
        for key, value in metric_list.items():
            metric_list[key] = sum(value) / len(value)
        return metric_list


# Loop with parameter grid search
# Parameters:
# - num_clients: Number of clients
# - num_rounds: Number of rounds of federated learning
# - num_local_epochs: Number of local epochs
# - data_distibution: Data distribution of the clients
    
# Optional:
# - model_learning_rate: Learning rate of the model
# - model_momentum: Momentum of the model
# - model_weight_decay: Weight decay of the model

rang_num_clients = range(2, 20, 2)
rang_num_rounds = range(1, 20, 2)
rang_num_local_epochs = range(1, 20, 2)

# Cross product of all parameters
param_grid = list(itertools.product(rang_num_clients, rang_num_rounds, rang_num_local_epochs))

print("Number of parameter combinations: {}".format(len(param_grid)))
for i, params in enumerate(param_grid):
    num_clients, num_rounds, num_local_epochs = params
    print("Parameter combination {}: {}".format(i, params))

    # Define parameter
    experiment_id = 1+i
    config = {
        'seed': 127491,
        'num_clients': num_clients,
        'num_rounds': num_rounds,
        'num_local_epochs': num_local_epochs,
        'perc_data': 1.0,
    }

    num_rounds = config["num_rounds"]
    num_clients = config["num_clients"]
    config["perc_data"] = 1.0 / num_clients

    # Set seed
    seed = config['seed']
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Create n clients
    # A client is a python script named flower_client.py, the following code block will create n clients
    # and start the training process.

    command = ".venv/bin/python flower_client.py --client_id {} --seed {} --perc_data {}"
    client_processes = []

    def run_client(client_id, command, config={}):
        client_command = command.format(client_id, config['seed'], config["perc_data"])
        process = subprocess.Popen(client_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        
        # Open the output file in write mode
        with open(f'client_{client_id}.txt', 'w') as file:
            # Print the output in real time and write it to the file
            for line in process.stdout:
                file.write(line)
                file.flush()
        
        # Wait for the process to finish
        process.wait()

    def run_server(experiment_id, num_rounds):
        # Define Flower strategy
        stategy = FashionTrainingStrategy(
            on_fit_config_fn=lambda _: config,
        )

        # Start Flower server
        hist = fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=stategy,
        )

        # Sav logs
        raw_metrics = { metric_name: [it[1] for it in values] for metric_name, values in hist.metrics_distributed.items() }
        save_experiment_data(experiment_id, config, raw_metrics)

        # Visualize training performance
        visualize_federated_learning_performance(
            experiment_id=experiment_id,
            cycles=num_rounds,
            average_train_accuracies=[it[1] for it in hist.metrics_distributed['accuracy']],
            average_train_losses=[it[1] for it in hist.metrics_distributed['loss']],
        )


    with concurrent.futures.ThreadPoolExecutor() as executor:
        executor.submit(run_server, experiment_id, num_rounds)
        time.sleep(2)

        for i in range(num_clients):
            future = executor.submit(run_client, i, command, config)
            client_processes.append(future)
            print("Client {} started".format(i))

    print("All clients finished")
