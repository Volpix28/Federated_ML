import itertools
import os
import random
import time
import requests
import torch
import flwr as fl
import subprocess
import concurrent.futures
import numpy as np
import pandas as pd
from logging import DEBUG, INFO
from typing import Callable, Dict, Optional, Tuple
from flwr.common.logger import log
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar

from utils import visualize_federated_learning_performance, save_experiment_data

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



# This variable is used to select the experiment to run
CURR_EXPERIMENT = 'experiment_4'

# List of experiments
experiments = {
    # Experiment 1
    #  Broad scan over the parameter space
    'experiment_1': {
        'rang_num_clients': range(2, 20, 2),
        'rang_num_rounds': range(1, 20, 2),
        'rang_num_local_epochs': range(1, 20, 2),
        'data_distribution': ['uniform']
    },
    # Experiment 2
    #  Same as experiment 1 but with less variance in the parameters
    #  Runtime:
    #    from 2024-01-21 14:48:00,000
    #    to   2024-01-21 16:48:35,634
    'experiment_2': {
        'rang_num_clients': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'rang_num_rounds': [5, 10],
        'rang_num_local_epochs': [10],
        'data_distribution': ['uniform']
    },
    # Experiment 3
    #  Changed minimal client number to num_clients
    'experiment_3': {
        'rang_num_clients': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'rang_num_rounds': [5, 10],
        'rang_num_local_epochs': [10],
        'data_distribution': ['uniform']
    },
    # Experiment 4
    #  Tested best parameters from all experiments
    'experiment_4': {
        'rang_num_clients': [5],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': ['uniform']
    },

    # Experiment 5
    #  
    'experiment_5': {
        'rang_num_clients': [2, 5, 10],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': ['uniform']
    },

    # Experiment 6
    # Data distribution testing of one client has 90% of the data
    'experiment_6': {
        'rang_num_clients': [3],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': [ 
            [ 0.8, 0.1, 0.1 ],
         ]
    },
    'experiment_7': {
        'rang_num_clients': [5],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': [ 
            [ 0.8, 0.05, 0.05, 0.05, 0.05 ],
        ]
    },
    'experiment_8': {
        'rang_num_clients': [5],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': [
            [ 0.3, 0.1, 0.1, 0.3, 0.2 ],
        ]
    },
    'experiment_9': {
        'rang_num_clients': [5],
        'rang_num_rounds': [10],
        'rang_num_local_epochs': [5],
        'data_distribution': [
            [ 0.2, 0.2, 0.2, 0.2, 0.2 ],
        ]
    },
}

# Select experiment to run
experiment_executed = [ 
    #*['experiment_6'for _ in range(5)],
    #*['experiment_7'for _ in range(5)],
    #*['experiment_8'for _ in range(5)],
    'experiment_4'
]
for experiment_name in experiment_executed:
    experiment = experiments[experiment_name]

    # Get experiment id from log file
    df = pd.read_csv('logs/experiments.csv', sep=';')
    start_experiment_id = df['id'].max() + 1
    del df

    # Cross product of all parameters
    param_grid = list(itertools.product(*experiment.values()))
    log(INFO, "Experiment: {}".format(CURR_EXPERIMENT))
    log(DEBUG, "Starting with experiment id: {}".format(start_experiment_id))
    log(DEBUG, "Number of parameter combinations: {}".format(len(param_grid)))
    for i, params in enumerate(param_grid):
        num_clients, num_rounds, num_local_epochs, data_distribution, *_ = params
        log(DEBUG, "Parameter combination {}: {}".format(i, params))

        # Define parameter
        experiment_id = start_experiment_id+i
        config = {
            'seed': 127491,
            'num_clients': num_clients,
            'num_rounds': num_rounds,
            'num_local_epochs': num_local_epochs,
            'data_distribution': data_distribution,
        }

        num_rounds = config["num_rounds"]
        num_clients = config["num_clients"]

        if config["data_distribution"] == 'uniform':
            data_distribution = [1.0 / num_clients for _ in range(num_clients)]
            config["data_distribution"] = data_distribution
        

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
        unbalanced_distribution = [(0.7, 0.9)]
        client_data_id_map = torch.randperm(num_clients)
        min_data_dist = (1 / num_clients) * 0.8 
        data_perc_sum = 0

        def run_client(client_id, command, config={}):
            # Resolve data distribution for this client
            perc_data = 1
            if type(data_distribution) == list:
                perc_data = 1

            elif data_distribution == 'unbalanced':
                client_data_id = client_data_id_map[client_id]
                data_dist = unbalanced_distribution[client_data_id] if client_data_id < len(unbalanced_distribution) else None
                data_range_needed_for_other_clients = (min_data_dist * (num_clients - client_id))
                rest_data_avaiable = (1-data_perc_sum)
                available_range = rest_data_avaiable - data_range_needed_for_other_clients
                # if not last client
                if data_dist:
                    perc_data = available_range * random.uniform(*data_dist)
                else:
                    perc_data = available_range * (random.random() if client_id < num_clients-1 else 1)
                data_perc_sum += perc_data

            # Start client
            client_command = command.format(client_id, config['seed'], perc_data)
            process = subprocess.Popen(client_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
            log(DEBUG, "Client {} started with command {}".format(i, client_command))

            # Open the output file in write mode
            os.makedirs('clients', exist_ok=True)
            with open(f'clients/client_{client_id}.txt', 'w') as file:
                # Print the output in real time and write it to the file
                for line in process.stdout:
                    file.write(line)
                    file.flush()
            
            # Wait for the process to finish
            process.wait()

        def run_server(experiment_id, num_rounds):
            # Define Flower strategy
            stategy = FashionTrainingStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
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


        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.submit(run_server, experiment_id, num_rounds)
            time.sleep(2)

            for i in range(num_clients):
                future = executor.submit(run_client, i, command, config)
                client_processes.append(future)

        log(INFO, "All clients finished")


# Push message to webhook
webhook = 'https://discord.com/api/webhooks/1203668220779827210/fXiR0Iy6v6ZqQX8opQeu--j0Fs9-U0Afrf7_wmZY7A7yTgio5LmLcAHxYNp_hkFr2IH7'
message = f"Experiment finished."
requests.post(webhook, json={"content": message})
