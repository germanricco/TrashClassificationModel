"""
Trains a PyTorch image classification model using device-agnostic code.
"""

# CORREGIR

import os
import torch
import data_setup, engine, utils


import torchvision
from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 4
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001
NUM_WORKERS = os.cpu_count()

# Setup directories
train_dir = "C:/Users/Germán/Documents/00 Proyectos/Greenly/trashClassification/data/train"
test_dir = "C:/Users/Germán/Documents/00 Proyectos/Greenly/trashClassification/data/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Parámetros del modelo con el que busco hacer Transfer Learning
weights = torchvision.models.MobileNet_V3_Large_Weights.DEFAULT
auto_transforms = weights.transforms()

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transforms=auto_transforms,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS
)

# Instancio el modelo
model = torchvision.models.mobilenet_v3_large(weights=weights)

# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
for param in model.features.parameters():
    param.requires_grad = False

# Get the length of class_names (one output unit for each class)
output_shape = len(class_names)
print(output_shape)

# Recreate the classifier layer and seed it to the target device
model.classifier[-1] = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2, inplace=True), 
    torch.nn.Linear(in_features=1280, 
                    out_features=output_shape, # same number of output units as our number of classes
                    bias=True)).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# # Save the model with help from utils.py
# utils.save_model(model=model,
#                  target_dir="models",
#                  model_name="mobilenetv3_large_v2.pth")
