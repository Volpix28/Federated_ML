import numpy as np
import copy
import model as my_model
import torch


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
