from collections import OrderedDict
import warnings

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
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


def train(net, trainloader, epochs):
  """Train the model on the training set."""
  criterion = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
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


def load_data():
  """Load FashionMNIST (training and test set)."""
  batch_size = 32
  trnasform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.1307,), (0.3081,))
  ])
  trainset = datasets.FashionMNIST('./dataset', train=True, download=True, transform=trnasform)
  valset = datasets.FashionMNIST('./dataset', train=False, download=True, transform=trnasform)
  train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
  val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True)
  return train_loader, val_loader

# #############################################################################
# Federating the pipeline with Flower
# #############################################################################


# Load model and data (simple CNN, CIFAR-10)
net = FashionSimpleNet().to(DEVICE)
trainloader, testloader = load_data()

# Define Flower client
class FlowerClient(fl.client.NumPyClient):
  def get_parameters(self, config):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

  def set_parameters(self, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

  def fit(self, parameters, config):
    self.set_parameters(parameters)
    train(net, trainloader, epochs=1)
    loss, accuracy = test(net, testloader)
    metrics = {
      'loss': float(loss),
      'accuracy': float(accuracy)
    }
    return self.get_parameters(config={}), len(trainloader), metrics

  def evaluate(self, parameters, config):
    self.set_parameters(parameters)
    loss, accuracy = test(net, testloader)
    print((loss, accuracy))
    metrics = {
      'loss': float(loss),
      'accuracy': float(accuracy)
    }
    return float(loss), len(trainloader), metrics
  
  def save(self, parameters, path):
    self.set_parameters(parameters)
    torch.save(net.state_dict(), path)

# Start Flower client
fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=FlowerClient())
