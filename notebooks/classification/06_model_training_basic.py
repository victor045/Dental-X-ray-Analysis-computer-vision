#!/usr/bin/env python
# coding: utf-8

# ## Model training: The toothmodel_basic.ToothModel class ##
# 
# The ToothModel class is a subclass of the LightningModule from PyTorch Lightning.
# 
# The class starts with the *\__init()__* method, which sets up the initial conditions for the class. These conditions include attributes like train_dataset, batch_size, num_workers, lr (learning rate), and model. In the case that no model is passed during initialization, a ResNet50Model is created. The class also sets up its loss function as cross entropy loss.
# 
# The *train_dataloader()* method returns a DataLoader object which represents the training dataset. The dataset is shuffled and loaded based on the batch size and number of workers defined during initialization.
# 
# The *forward()* method performs a forward pass through the model and returns the output.
# 
# The *training_step()* method performs a forward pass as well, but with the additional step of calculating loss between the predictions and actual values.
# 
# The *predict_step()* method is also similar to the forward() method, but it is used during the prediction phase and hence doesn't involve computing loss.
# 
# The *configure_optimizers()* method sets the optimizer for the model. In this case, AdamW is used.

# Imports
import os
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
import glob

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# PyTorch packages
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch import Trainer

# Albumentations library
import albumentations as alb

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.imageproc import ImageData
from dentexmodel.torchdataset import DatasetFromDF, load_and_process_image
print(f'Dentexmodel package version:  {dm.__version__}')
print(f'PyTorch version:              {torch.__version__}')
print(f'PyTorch Lightning version:    {pl.__version__}')

# GPU checks
is_cuda = torch.cuda.is_available()
print(f'CUDA available: {is_cuda}')
print(f'Number of GPUs found:  {torch.cuda.device_count()}')

if is_cuda:
    print(f'Current device ID:     {torch.cuda.current_device()}')
    print(f'GPU device name:       {torch.cuda.get_device_name(0)}')
    print(f'CUDNN version:         {torch.backends.cudnn.version()}')
    device_str = 'cuda:0'
    torch.cuda.empty_cache() 
else:
    device_str = 'cpu'
device = torch.device(device_str)
print()
print(f'Device for model training/inference: {device}')

# Path settings 
# Main data directory (defined as environment variable in docker-compose.yml)
data_root = os.environ.get('DATA_ROOT')

# Download directory (change as needed)
dentex_dir = os.path.join(data_root, 'dentex')
model_dir = os.path.join(data_root, 'model')
data_dir = os.path.join(dentex_dir, 'dentex_classification')

# This image directory is where the xrays are in the archive, so should be left as-is
image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays')
cropped_image_dir = os.path.join(image_dir, 'crop')

# Directory for the output
output_dir = os.path.join(data_dir, 'output')

data_file_name = 'dentex_disease_datasplit.parquet'
data_file = os.path.join(data_dir, data_file_name)

# ### Create PyTorch datasets from data frame ###

data_df = pd.read_parquet(data_file)
# Convert class names to labels
cl_names = sorted(list(data_df['label'].unique()))
# Let's assign number to the classes
cl_numbers = [data_df.loc[data_df['label'] == label, 'cl'].values[0] for label in cl_names]
label_dict = dict(zip(cl_names, cl_numbers))
cl_dict = dict(zip(cl_numbers, cl_names))
# Show the class labels
print(f"{content}")
# For this model, we will use the training and testing data sets. 
# We will keep the validation set for the 'fancy' version of the model
train_df = data_df.loc[data_df['dataset'] == 'train']
test_df = data_df.loc[data_df['dataset'] == 'test']
n_train_samples = len(train_df['box_name'].unique())
n_test_samples = len(test_df['box_name'].unique())
print(f'Using {n_train_samples} samples for training.')
print(f'Using {n_test_samples} samples for testing.')

# Augmentations
# Image augmentations is part of the PyTorch dataset

# The output of this transformation must match the required input size for the model
max_image_size = 550
im_size = 224

# Definition of the image augmentations for the training set
train_transform = alb.Compose([
    alb.Resize(im_size + 32, im_size + 32),
    alb.RandomCrop(im_size, im_size),
    alb.HorizontalFlip(),
    alb.ShiftScaleRotate(),
    alb.Blur(),
    alb.RandomGamma(),
    alb.Sharpen(),
    alb.GaussNoise(),
    alb.CoarseDropout(16, 32, 32),
    alb.CLAHE(),
    alb.Normalize(mean=ImageData().image_net_mean, 
                  std=ImageData().image_net_std)])

# For testing, we do not want any augmentations
# but we will still need the correct input size and image normalization
val_transform = alb.Compose([
    alb.Resize(im_size, im_size),
    alb.Normalize(mean=ImageData().image_net_mean, 
                  std=ImageData().image_net_std)])

# Create the data sets from the data frame
train_dataset = DatasetFromDF(data=train_df,
                              file_col='box_file',
                              label_col='cl',
                              max_image_size=max_image_size,
                              transform=train_transform,
                              validate=True)

test_dataset = DatasetFromDF(data=test_df,
                             file_col='box_file',
                             label_col='cl',
                             max_image_size=max_image_size,
                             transform=val_transform,
                             validate=True)

# ### The ResNet50Model class ###
# The ResNet50Model class implements a variation of the ResNet50 architecture, which is a well-known type of convolutional neural networks particularly suitable for image classification tasks.
# 
# The *\__init__()* method initializes the instance of this class with the number of outputs.
# The *create_model()* method creates a ResNet50 model with default weights. The last fully Connected layer model.fc of the model is replaced with a sequential arrangement of layers combining a Linear layer, a ReLU activation function, and another Linear layer.
# 
# The first Linear layer transforms the input features to 512 dimensions. The output of this layer is then passed through a ReLU activation function. Finally, the output is transformed by the second Linear layer to match the number of output classes(self.n_outputs).
# 
# A list of available models is here:
# https://pytorch.org/vision/stable/models.html#classification
# 
# The ResNet50 model:
# https://pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html#torchvision.models.resnet50

from torchvision.models import resnet50, ResNet50_Weights

class ResNet50Model:
    """ This is the ResNet50 model from torchvision.models """
    def __init__(self, n_outputs=4):
        self.n_outputs = n_outputs

    def create_model(self):
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        model.fc = nn.Sequential(
            nn.Linear(in_features=model.fc.in_features, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=self.n_outputs)
        )
        return model

# toothmodel1 is a minimal Lightning model to train
# there is no trainig or validation metrics, just the bar minimum
from dentexmodel.models.toothmodel_basic import ToothModel
model = ToothModel(train_dataset=train_dataset,
                   batch_size=16,
                   num_workers=2,
                   model=ResNet50Model(n_outputs=4).create_model())

# ### Test the model output ###

# Run one batch of images through the model
dl = model.train_dataloader()
image_batch, label_batch = next(iter(dl))
print(image_batch.numpy().shape)
print(label_batch.numpy().shape)

# ### Train the model ###
# Training on GPU is recommended. Training works on a CPU-only machine, but it is very slow.

# Create the trainer object and train the model for 5 epochs
# Train for at least 40 epochs to get good results

max_epochs = 5
tr = Trainer(max_epochs=max_epochs,
             deterministic=True,
             default_root_dir=model_dir)
# Run the training
tr.fit(model)
