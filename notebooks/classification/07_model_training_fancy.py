#!/usr/bin/env python
# coding: utf-8

# ## Advanced model with learning rate scheduler and performance metrics ##

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
from lightning.pytorch.loggers import TensorBoardLogger
import lightning.pytorch as pl
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateFinder, LearningRateMonitor
import torchmetrics

# Albumentations library
import albumentations as alb

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData
from dentexmodel.models.toothmodel_fancy import ToothModel, FineTuneLearningRateFinder
from dentexmodel.torchdataset import DatasetFromDF, load_and_process_image
print(f'dentexmodel package version:  {dm.__version__}')

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
# Get the class labels
cl_numbers = [data_df.loc[data_df['label'] == label, 'cl'].values[0] for label in cl_names]
label_dict = dict(zip(cl_names, cl_numbers))
cl_dict = dict(zip(cl_numbers, cl_names))
# Show the class labels
print(f"{content}")
# Select the samples for training, validation and testing from our data frame
train_df = data_df.loc[data_df['dataset']=='train']
val_df = data_df.loc[data_df['dataset']=='val']
test_df = data_df.loc[data_df['dataset']=='test']

train_samples = sorted(list(train_df['box_name'].unique()))
print(f'Found {len(train_samples)} samples in the training set.')
val_samples = sorted(list(val_df['box_name'].unique()))
print(f'Found {len(val_samples)} samples in the validation set.')
test_samples = sorted(list(test_df['box_name'].unique()))
print(f'Found {len(test_samples)} samples in the test set.')
print()

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

# For validation and testing, we do not want any augmentations
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

val_dataset = DatasetFromDF(data=val_df,
                            file_col='box_file',
                            label_col='cl',
                            max_image_size=max_image_size,
                            transform=val_transform,
                            validate=True)

test_dataset = DatasetFromDF(data=test_df,
                             file_col='box_file',
                             label_col='cl',
                             max_image_size=max_image_size,
                             transform=val_transform,
                             validate=True)

# ### Training the model with learning rate scheduling ###

# Model parameters and name
seed = 234
model_name = 'FancyLR'
model_version = 1
# Train for 40 epochs to get good results
max_epochs = 80
num_classes = 4
num_workers = 2
batch_size = 16
initial_lr = 1.0e-3
check_val_every_n_epoch = 1
checkpoint_every_n_epoch = 2
save_top_k = 3

# Create the model
model = ToothModel(train_dataset=train_dataset,
                   val_dataset=val_dataset,
                   test_dataset=test_dataset,
                   batch_size=batch_size,
                   num_classes=num_classes,
                   num_workers=num_workers,
                   lr=initial_lr)

# Setup logger
logger = TensorBoardLogger(save_dir=model_dir,
                           name=model_name,
                           version=model_version)

# Checkpoint callback
checkpoint_dir = os.path.join(model_dir, 
                              model_name,
                              f'version_{model_version}',
                              'checkpoints')

Path(checkpoint_dir).mkdir(exist_ok=True, parents=True)
chk_callback = ModelCheckpoint(dirpath=checkpoint_dir,
                               filename='dentexmodel-{epoch}',
                               monitor='val_loss',
                               mode='min',
                               save_last=True,
                               every_n_epochs=checkpoint_every_n_epoch,
                               save_on_train_epoch_end=True,
                               save_top_k=save_top_k)

lr_finder = FineTuneLearningRateFinder(milestones=(5, 10), 
                                       min_lr=1.0e-8,  
                                       max_lr=0.01, 
                                       num_training_steps=100,
                                       mode='exponential',
                                       early_stop_threshold=None,
                                       update_attr=True)

lr_starter = LearningRateFinder(min_lr=1.0e-8,  
                                max_lr=0.01, 
                                num_training_steps=300,
                                mode='exponential',
                                early_stop_threshold=None,
                                update_attr=True)

lr_monitor = LearningRateMonitor(logging_interval='epoch',
                                 log_momentum=True)

print(f'Training the "{model_name}" model for {max_epochs} epochs.')
print()

tr = Trainer(max_epochs=max_epochs,
             default_root_dir=model_dir,
             callbacks=[chk_callback, lr_finder, lr_monitor],
             logger=logger,
             check_val_every_n_epoch=check_val_every_n_epoch)
tr.fit(model)
