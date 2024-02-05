from ast import Dict
from collections import OrderedDict
import argparse
import json
from typing import Dict, Iterator, Optional, Sequence
import warnings

import flwr as fl
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, Sampler
from torchvision.datasets import CIFAR10
from collections import OrderedDict

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


# Get seed from command-line argument
# Create argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--client_id', type=int, default=0, help="Client ID")
parser.add_argument("--seed", type=int, default=42, help="Seed value")
parser.add_argument('--perc_data', type=float, default=1, help="Percentage of data to use")
parser.add_argument('--class_distibution', type=json.loads, default=None, help="Class distribution of the data") # type: Dict[str, float]

# Parse command-line arguments
args = parser.parse_args()

# Set seed for random, numpy, pytorch
seed = args.seed
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
if is_xpu_available():
  torch.xpu.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# #############################################################################
# Regular PyTorch pipeline: nn.Module, train, test, and DataLoader
# #############################################################################

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = get_device()


class FashionSimpleNet(nn.Module):

    """ Simple network"""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # 28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2) # 7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 7 * 7)
        x = self.classifier(x)
        return x


def train(net, trainloader, epochs, lr=0.001, momentum=0.9):
  """Train the model on the training set."""
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
  for _ in range(epochs):
    for images, labels in trainloader:
      images, labels = images.to(DEVICE), labels.to(DEVICE)
      optimizer.zero_grad()
      loss = criterion(net(images), labels)
      loss.backward()
      optimizer.step()


def test(net, testloader):
  """Validate the model on the test set."""
  criterion = torch.nn.CrossEntropyLoss()
  correct, total, loss = 0, 0, 0.0
  with torch.no_grad():
    for data in testloader:
      images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
      outputs = net(images)
      loss += criterion(outputs, labels).item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
  accuracy = correct / total
  return loss, accuracy

def evaluate(net, testloader, class_distribution: Optional[Dict[str, int]] = None):
  """Validate the model on the test set."""
  criterion = torch.nn.CrossEntropyLoss()
  class_wise_total = get_class_distribution(testloader.dataset) if class_distribution is None else class_distribution
  correct, total, loss = 0, 0, 0.0
  class_wise_accuracy = {}
  with torch.no_grad():
    for data in testloader:
      images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
      outputs = net(images)
      loss += criterion(outputs, labels).item()
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)

      correct_tensor = (predicted == labels)
      correct_indices = correct_tensor.nonzero()
      for i in correct_indices:
        label = str(labels[i].item())
        class_wise_accuracy[label] = class_wise_accuracy.get(label, 0) + 1
      correct += correct_tensor.sum().item()
  
  class_wise_accuracy = {k: v/class_wise_total[k] for k, v in class_wise_accuracy.items()}
  accuracy = correct / total
  return loss, accuracy, class_wise_accuracy


def get_class_distribution(dataset):
  class_distribution = {}
  for it in dataset:
    label = str(it[1])
    if label not in class_distribution:
      class_distribution[label] = 0
    class_distribution[label] += 1
  return class_distribution

def sampled_indices_for_a_given_distribution(dataset, class_distribution: Dict[str, float]):
  # Gather class distribution
  class_indices = {}
  for i, it in enumerate(dataset):
      label = str(it[1])
      if label not in class_indices:
          class_indices[label] = []
      class_indices[label].append(i)
  class_indices = {k: torch.tensor(v) for k, v in class_indices.items()}

  for c in class_distribution:
    if c not in class_indices:
      raise ValueError(f"Class {c} not in the dataset.")

  sampled_per_class = { k: v[torch.randperm(len(v))[:(int(len(v)*class_distribution[k]))]] for k, v in class_indices.items() }
  return torch.cat([v for v in sampled_per_class.values()])


def load_data(client_id: int, percent: Optional[float] = 1.0, class_distribution: Optional[dict] = None):
  """
  Load FashionMNIST (training and test set).
  :param client_id: ID of the client
  :param percent: Percentage of the training data to use (between 0 and 1)
  """
  batch_size = 32
  tf = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
  trainset = datasets.FashionMNIST('./dataset', train=True, download=True, transform=tf)
  valset = datasets.FashionMNIST('./dataset', train=False, download=True, transform=tf)

  samples = len(trainset)
  print(f"Total number of samples: {samples}")
  print(f"Percentage of samples: {percent}")
  print(f"Class distribution: {class_distribution}")

  if class_distribution is None:
     sample_indices = torch.randperm(len(trainset))[:(int(samples*percent))]
  else:
     sample_indices = sampled_indices_for_a_given_distribution(trainset, class_distribution)
  train_sampler = SubsetRandomSampler(sample_indices)

  # Only distributed the training data not the validation data.
  train_loader = DataLoader(trainset, batch_size=batch_size, sampler=train_sampler)
  val_loader = DataLoader(valset, batch_size=batch_size)
  return train_loader, val_loader

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################


# Load model and data (simple CNN, CIFAR-10)
net = FashionSimpleNet().to(DEVICE)
trainloader, testloader = load_data(args.client_id, args.perc_data, args.class_distibution)
trainingset_class_distribution = get_class_distribution(trainloader.dataset)

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    print(config)
    self.set_parameters(parameters)
    train(net, trainloader, epochs=config['num_local_epochs'])
    loss, accuracy = test(net, testloader)
    metrics = {
      'loss': float(loss),
      'accuracy': float(accuracy)
    }
    return self.get_parameters(config={}), len(trainloader), metrics

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy, class_wise_accuracy = evaluate(net, testloader, class_distribution=trainingset_class_distribution)
    metrics = {
      'loss': float(loss),
      'accuracy': float(accuracy),
      'class_distribution': json.dumps(trainingset_class_distribution),
      'class_wise_accuracy': json.dumps(class_wise_accuracy),
    }
    return float(loss), len(trainloader), metrics
  
  def save(self, parameters, path):
    self.set_parameters(parameters)
    torch.save(net.state_dict(), path)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
