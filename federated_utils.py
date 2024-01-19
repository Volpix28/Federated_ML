import copy
from model_utils import get_model_weights, set_model_weights

def federated_averaging(models):
    # Federated averaging. Aggregate the weights of different models.
    global_weights = copy.deepcopy(get_model_weights(models[0]))
    for layer_weights in global_weights:
        layer_weights *= 0  
    
    # Sum weights from all models
    for model in models:
        local_weights = get_model_weights(model)
        for i, layer_weights in enumerate(local_weights):
            global_weights[i] += layer_weights
            
    # Average weights
    for layer_weights in global_weights:
        layer_weights /= len(models)
    
    return global_weights

def train_on_clients(client_models, client_data, local_epochs):
    # Perform training on each client and get their model weights.
    for client_idx, client_model in enumerate(client_models):
        data = client_data[client_idx]
        client_model.fit(data['x_train'], data['y_train'], epochs=local_epochs)
        
        # Evaluation needed?
