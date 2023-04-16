import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.resnet import ResNet, BasicBlock
import time
import psutil
import matplotlib.pyplot as plt
import os
import statistics

train_losses = []
test_accs = []
training_times = []
inference_times = []
gpu_utilization = []
batch_size = 16

if not os.path.exists("./results"):
  os.mkdir("./results")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Define the ResNet18 model
class ResNet18(ResNet):
  def __init__(self):
    super(ResNet18, self).__init__(block=models.resnet.BasicBlock,
                                   layers=[2, 2, 2, 2],
                                   num_classes=10)
    self.model = models.resnet18(pretrained=True)
    num_ftrs = self.model.fc.in_features
    self.model.fc = nn.Linear(num_ftrs, 10)  # Replace last fc layer with new one

  def forward(self, x):
    x = self.model(x)
    return x



class AlexNet(nn.Module):
  def __init__(self):
    super(AlexNet, self).__init__()
    self.model = models.alexnet(pretrained=True)
    num_ftrs = self.model.classifier[6].in_features
    self.model.classifier[6] = nn.Linear(num_ftrs, 10)  # Replace last fc layer with new one

  def forward(self, x):
    x = self.model(x)
    return x


class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()
    self.model = models.vgg16(pretrained=True)
    num_ftrs = self.model.classifier[6].in_features
    self.model.classifier[6] = nn.Linear(num_ftrs, 10)  # Replace last fc layer with new one

  def forward(self, x):
    x = self.model(x)
    return x


transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load the CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

# Initialize the ResNet18 model
# Define models
models_list = [VGG(), AlexNet(), ResNet18()]

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Initialize lists to store results
train_losses_list = []
test_accs_list = []
training_times_list = []
inference_times_list = []
gpu_utilization_list = []

epochs = 10
number_of_runs = 2

# Iterate over models
for i, model in enumerate(models_list):
  train_losses = []
  test_accs = []
  training_times = []
  inference_times = []
  gpu_utilization = []
  for run in range(number_of_runs):
    print(f"\nTraining model {i + 1}/{len(models_list)}: {type(model).__name__}")

    # Move model to device
    net = model.to(device)
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    # Initialize lists to store results for current model
    training_time = 0
    for epoch in range(epochs):
      net.train()
      running_loss = 0.0
      start_time = time.time()
      for data in trainloader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

      train_losses.append(running_loss / len(trainloader))
      training_time += (time.time() - start_time)
    training_times.append(training_time)
    # Testing starts
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
      for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    test_acc = (correct / total) * 100
    test_accs.append(test_acc)
    torch.save(net.state_dict(), "resnet18_fine_tuned_model.pth")
  print(f"\nEND: Training and testing model {i + 1}/{len(models_list)}: {type(model).__name__}")
  print("Mean training time: {}, Std dev: {}".format(sum(training_times) / len(training_times),
                                                     statistics.stdev(training_times)))
  print("Mean acc: {}, Std dev: {}".format(sum(test_accs) / len(training_times),
                                                     statistics.stdev(test_accs)))
