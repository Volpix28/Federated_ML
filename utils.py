import os
from typing import Sequence
import numpy as np
import copy
import model as my_model
import torch
import matplotlib.pyplot as plt

def create_new_model():
    # Implement model creation logic
    return my_model.FashionSimpleNet()

def load_model(model_path: str):
    """
    Load model from model_path
    :param model_path: path to model
    """
    model = create_new_model()
    try:
        model_state_dict = torch.load(model_path)
        model.load_state_dict(model_state_dict)
    finally:
        return model

def save_model(model, save_path):
    # Implement model saving logic
    torch.save(model.state_dict(), save_path)



def save_experiment_data(experiment_id: int, config: dict, metrics: dict):
    """
    Saves the data of an experiment into a csv file named experiments.csv, which contains columns for the experiment id, every parameter as well as metric.
    :param experiment_id: ID of the experiment
    :param config: Dictionary of parameters
    :param metrics: Dictionary of metrics
    """

    # Create csv file if it does not exist
    log_file = 'logs/experiments.csv'
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write('id;')
            f.write(';'.join(config.keys()))
            f.write(';')
            f.write(';'.join(metrics.keys()))
            f.write('\n')

    # Append experiment data to csv file
    with open(log_file, 'a') as f:
        f.write(str(experiment_id) + ';')
        f.write(';'.join([str(value) for value in config.values()]))
        f.write(';')
        f.write(';'.join([str(value) for value in metrics.values()]))
        f.write('\n')



def visualize_federated_learning_performance(experiment_id: int, cycles: int, average_train_accuracies: Sequence[float], average_train_losses: Sequence[float]):
    """
    Visualize the performance of the federated learning model over the cycles.
    
    :param cycles: Number of federated learning cycles
    :param average_train_accuracies: List of average training accuracies per cycle
    :param average_train_losses: List of average training losses per cycle
    """
    
    # Plot Training Loss and Accuracy
    plt.figure(figsize=(12, 5))

    # Plot average training accuracy
    plt.subplot(1, 2, 1)
    plt.plot(range(1, cycles + 1), average_train_accuracies, marker='o', color='b', label='Average Training Accuracy')
    plt.title(f'Experiment {experiment_id}: Average Training Accuracy per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy')
    plt.xticks(range(1, cycles + 1))  # Set x-axis ticks to integer values
    plt.grid(True)
    plt.legend()

    # Plot average training loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, cycles + 1), average_train_losses, marker='o', color='r', label='Average Training Loss')
    plt.title(f'Experiment {experiment_id}: Average Training Loss per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Loss')
    plt.xticks(range(1, cycles + 1))  # Set x-axis ticks to integer values
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    # save as png file
    out_path = f'logs/figures'
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f'{out_path}/experiment_{experiment_id}.png')
