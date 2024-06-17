"""
import torchvision

model = torchvision.models.resnet18()
print(model.device)

"""
import torch
import torchvision

# Instantiate the model
model = torchvision.models.resnet18()

# Get the device of the model's parameters
device = next(model.parameters()).device
print(device)
