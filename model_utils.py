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

def get_model_weights(model):
    # Retrieve model weights
    return model.get_weights()

def set_model_weights(model, weights):
    # Set model weights
    model.set_weights(weights)

def visualize_federated_learning_performance(cycles, average_train_accuracies, average_train_losses):
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
    plt.title('Average Training Accuracy per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()

    # Plot average training loss
    plt.subplot(1, 2, 2)
    plt.plot(range(1, cycles + 1), average_train_losses, marker='o', color='r', label='Average Training Loss')
    plt.title('Average Training Loss per Cycle')
    plt.xlabel('Cycle')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()
