import itertools
import json
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
from logging import DEBUG, INFO, WARNING
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from flwr.common import EvaluateRes
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
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
        server_config: Dict[str, Any] = {}
    ) -> None:
        self.server_config = server_config
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

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg([ (evaluate_res.num_examples, evaluate_res.loss) for _, evaluate_res in results ])

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics, server_round)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")
        return loss_aggregated, metrics_aggregated
    

    def evaluate_metrics_aggregation_fn(self, evaluate_res, server_round: Optional[int] = None) -> Dict[str, Any]:
        """
        Aggregate the evaluation metrics.
        :param evaluate_res: List of evaluation results from clients. Each element is a tuple (loss, accuracy, ...).
        :return: Aggregated evaluation result.
        """
        max_round = self.server_config.get('num_rounds', 1)
        skip_metrics = [ 'class_distribution', 'class_wise_accuracy' ]

        metric_list = {}
        for _, metrics in evaluate_res:
            for key, value in metrics.items():
                if key in skip_metrics: continue
                if key not in metric_list:
                    metric_list[key] = []
                metric_list[key].append(value)
        
        # Compute mean loss and accuracy
        for key, value in metric_list.items():
            metric_list[key] = sum(value) / len(value)

        # Evaluate class distribution accuracy
        class_wise_accuracy = {}
        for _, metrics in evaluate_res:
            class_wise_accuracy_client = json.loads(metrics['class_wise_accuracy'])
            for key, value in class_wise_accuracy_client.items():
                if key not in class_wise_accuracy:
                    class_wise_accuracy[key] = []
                class_wise_accuracy[key].append(value)
        
        # Compute mean class wise accuracy
        for key, value in class_wise_accuracy.items():
            class_wise_accuracy[key] = sum(value) / len(value)
        metric_list['class_wise_accuracy'] = class_wise_accuracy

        if server_round == max_round:
            metric_list['class_distribution'] = [ v['class_distribution'] for _, v in evaluate_res ]
        
        #metric_list['class_distribution'] = {}
        #metric_list['class_wise_accuracy'] = {}
        #TODO debug and save all class wise accuracy
        #print()

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
        'num_clients': range(2, 20, 2),
        'num_rounds': range(1, 20, 2),
        'num_local_epochs': range(1, 20, 2),
        'data_distribution': ['uniform'],
        'class_distibution': ['uniform']
    },
    # Experiment 2
    #  Same as experiment 1 but with less variance in the parameters
    #  Runtime:
    #    from 2024-01-21 14:48:00,000
    #    to   2024-01-21 16:48:35,634
    'experiment_2': {
        'num_clients': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'num_rounds': [5, 10],
        'num_local_epochs': [10],
        'data_distribution': ['uniform'],
        'class_distibution': ['uniform']
    },
    # Experiment 3
    #  Changed minimal client number to num_clients
    'experiment_3': {
        'num_clients': [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'num_rounds': [5, 10],
        'num_local_epochs': [10],
        'data_distribution': ['uniform'],
        'class_distibution': ['uniform']
    },
    # Experiment 4
    #  Tested best parameters from all experiments
    'experiment_4': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': ['uniform'],
        'class_distibution': ['uniform']
    },

    # Experiment 5
    #  
    'experiment_5': {
        'num_clients': [2, 5, 10],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': ['uniform'],
        'class_distibution': ['uniform']
    },

    # Experiment 6
    # Data distribution testing of one client has 90% of the data
    'experiment_6': {
        'num_clients': [3],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': [ 
            [ 0.8, 0.1, 0.1 ],
         ],
        'class_distibution': ['uniform']
    },
    'experiment_7': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': [ 
            [ 0.8, 0.05, 0.05, 0.05, 0.05 ],
        ],
        'class_distibution': ['uniform']
    },
    'experiment_8': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': [
            [ 0.3, 0.1, 0.1, 0.3, 0.2 ],
        ],
        'class_distibution': ['uniform']
    },
    'experiment_9': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': [
            [ 0.2, 0.2, 0.2, 0.2, 0.2 ],
        ],
        'class_distibution': ['uniform']
    },

    'experiment_10': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': ['uniform'],
        'class_distibution': [ 
            [
                { '0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1 },
                { '0': 0.1, '1': 0.9, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1 },
                { '0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.9, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1 },
                { '0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.9, '9': 0.1 },
                { '0': 0.1, '1': 0.1, '2': 0.1, '3': 0.1, '4': 0.1, '5': 0.1, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1 },
            ]
        ]
    },
    'experiment_11': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': ['uniform'],
        'class_distibution': [ 
            [
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0, '9': 0.0 },
                { '0': 0.0, '1': 1.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0, '9': 0.0 },
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 1.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0, '9': 0.0 },
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 1.0, '9': 0.0 },
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.0, '9': 0.0 },
            ]
        ]
    },
    'experiment_12': {
        'num_clients': [5],
        'num_rounds': [10],
        'num_local_epochs': [5],
        'data_distribution': ['uniform'],
        'class_distibution': [ 
            [
                { '0': 0.1, '1': 0.1, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.1, '9': 0.1 },
                { '0': 0.3, '1': 0.3, '2': 0.1, '3': 0.1, '4': 0.0, '5': 0.0, '6': 0.0, '7': 0.0, '8': 0.1, '9': 0.1 },
                { '0': 0.0, '1': 0.0, '2': 0.3, '3': 0.3, '4': 0.1, '5': 0.1, '6': 0.0, '7': 0.0, '8': 0.1, '9': 0.1 },
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.3, '5': 0.3, '6': 0.1, '7': 0.1, '8': 0.1, '9': 0.1 },
                { '0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0, '5': 0.0, '6': 0.3, '7': 0.3, '8': 0.1, '9': 0.1 },
            ]
        ]
    },
}

config_template = {
    'seed': 127491,
}

# Select experiment to run
experiment_executed = [
    *['experiment_4' for _ in range(3)],
    *['experiment_10' for _ in range(5)],
    *['experiment_11' for _ in range(5)],
    *['experiment_12' for _ in range(5)],
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
    start_time = time.time()
    for i, params in enumerate(param_grid):
        # Define parameter
        config = config_template.copy()
        config.update(dict(zip(list(experiment.keys()), params)))
        experiment_id = start_experiment_id+i
        num_clients = config["num_clients"]
        
        # Log parameter combination
        log(DEBUG, "Parameter combination {}: {}".format(i, config))

        # Set dynamic parameters
        if config["data_distribution"] == 'uniform':
            data_distribution = [1.0 / num_clients for _ in range(num_clients)]
            config["data_distribution"] = data_distribution
        if config["class_distibution"] == 'uniform':
            class_distibution = [None for _ in range(num_clients)]
            config["class_distibution"] = class_distibution
            
        # unpack parameters
        seed, num_clients, num_rounds, num_local_epochs, data_distribution, class_distibution, *_ = config.values()

        # Set seed
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



        command = ".venv/bin/python flower_client.py --client_id {} --seed {} --perc_data {} --class_distibution '{}'"
        client_processes = []
        unbalanced_distribution = [(0.7, 0.9)]
        min_data_dist = (1 / num_clients) * 0.8
        client_data_id_map = torch.randperm(num_clients)
        data_perc_sum = 0

        def run_client(client_id, command, config={}):
            # Resolve data distribution for this client
            client_data_id = client_data_id_map[client_id].item()
            perc_data = 1
            if type(data_distribution) == list:
                perc_data = data_distribution[client_data_id]

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

            # Data Class distribution
            dist = class_distibution
            if type(class_distibution) == list:
                dist = class_distibution[client_data_id]

            # Start client
            client_command = command.format(client_id, config['seed'], perc_data, json.dumps(dist))
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
            attr_filter = [ 'seed', 'num_clients', 'num_rounds', 'num_local_epochs' ]
            server_config = config.copy()
            for it in list(server_config.keys()):
                if not it in attr_filter:
                    del server_config[it]

            # Define Flower strategy
            stategy = FashionTrainingStrategy(
                min_fit_clients=num_clients,
                min_evaluate_clients=num_clients,
                min_available_clients=num_clients,
                on_fit_config_fn=lambda _: server_config,
                server_config=server_config,
            )

            # Start Flower server
            hist = fl.server.start_server(
                server_address="0.0.0.0:8080",
                config=fl.server.ServerConfig(num_rounds=num_rounds),
                strategy=stategy,
            )
            end_time = time.time()

            # Sav logs
            raw_metrics = { metric_name: [it[1] for it in values] for metric_name, values in hist.metrics_distributed.items() }
            save_experiment_data(experiment_id, {
                'experiment_name': experiment_name,
                'start_time': start_time, 
                'end_time': end_time,
                **config
            }, raw_metrics)

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
