from typing import Callable, Dict, Optional, Tuple
import flwr as fl
from flwr.common import MetricsAggregationFn, NDArrays, Parameters, Scalar


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
        evaluate_metrics_aggregation_fn: MetricsAggregationFn | None = None
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

    def evaluate_metrics_aggregation_fn(evaluate_res):
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
# - num_rounds: Number of rounds of federated learning
# - num_clients: Number of clients
# - model_learning_rate: Learning rate of the model
# - model_momentum: Momentum of the model
# - model_weight_decay: Weight decay of the model

# Define parameter
num_rounds = 5

# Define Flower strategy
stategy = FashionTrainingStrategy()

# Start Flower server
hist = fl.server.start_server(
  server_address="0.0.0.0:8080",
  config=fl.server.ServerConfig(num_rounds=num_rounds),
  strategy=stategy,
)

hist.metrics_distributed # { 'loss': [], 'accuracy': [] }
