#!/usr/bin/env python
# coding: utf-8

# ## Dataloaders ##

# Imports
import os
import numpy as np
import pandas as pd
import cv2

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# PyTorch methods
import torch
from torch.utils.data import DataLoader

# Albumentations library
import albumentations as alb

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.imageproc import ImageData
from dentexmodel.torchdataset import DatasetFromDF, load_and_process_image

print(f'Project module version: {dm.__version__}')

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

# ### Create PyTorch dataset from data frame ###

data_df = pd.read_parquet(data_file)
label_list = sorted(list(data_df['label'].unique()))
cl_list = [data_df.loc[data_df['label']==label, 'cl'].values[0] for label in label_list]
cl_dict = dict(zip(cl_list, label_list))
print(f"{content}")
print()
print(f"{content}")
# Maximum image size
max_im_width = data_df['im_width'].max()
max_im_height = data_df['im_height'].max()

print(f'Maximum image height across the data set: {max_im_height}')
print(f'Maximum image width across the data set:  {max_im_width}')

# The maximum dimension is the max_im_height:
max_image_size = np.max([max_im_height, max_im_width])

# ### PyTorch dataset from data frame ###
# Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled from our model training code for better readability and modularity. PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
# 
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html

# Image augmentations is part of the PyTorch dataset

# The output of this transformation must match the required input size for the model
im_size = 512

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
    alb.Normalize(mean=0, std=1)])

# Vor validation and testing, we do not want any augmentations
# but we will still need the correct input size and image normalization
val_transform = alb.Compose([
    alb.Resize(im_size, im_size),
    alb.Normalize(mean=0, std=1)])

train_df = data_df.loc[data_df['dataset'] == 'train']
print(f"{content}")
# Create the data sets from the data frame
train_dataset = DatasetFromDF(data=train_df,
                              file_col='box_file',
                              label_col='cl',
                              max_image_size=max_image_size,
                              transform=train_transform,
                              validate=True)

# ### Retrieve one image from the data set ###
# Everytime the cell is run, a difference augmentation is generated from the same image

def image_stats(ig, decimals=3):

    """ Returns pandas series with image stats """
    
    output_dict = {'im_width': ig.shape[1],
                   'im_height': ig.shape[0],
                   'im_min': np.round(np.min(ig), decimals=decimals),
                   'im_max': np.round(np.max(ig), decimals=decimals),
                   'im_mean': np.round(np.mean(ig), decimals=decimals),
                   'im_std': np.round(np.std(ig), decimals=decimals)}
                   
    return pd.DataFrame(output_dict, index=[0]).iloc[0]

# Retrieve one (image, label) sample from the data set
image_index = 234
image_sample, label_sample = train_dataset[image_index]
# We need to move the color channel back to the end of the array
image = np.transpose(image_sample.numpy(), (1, 2, 0))
label = cl_dict.get(int(label_sample.numpy()))
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(image)
ax.set(xticks=[], yticks=[], title=str(label))
plt.show()
print()
print(f"{content}")
# ### PyTorch DataLoader ###
# In PyTorch, a DataLoader is part of the torch.utils.data module which is used for loading datasets in a systematic and organized way.
# The data loader takes in a dataset and transforms it into batches so that the data can be processed more efficiently.

dataloader = DataLoader(dataset=train_dataset, batch_size=16)

# The dataloader can be converted into an iterator which returns images and labels
image_batch, label_batch = next(iter(dataloader))

print(f'Size of the image batch: {image_batch.numpy().shape}')
print(f'Labels:                  {label_batch.numpy()}')

# ### DataLoader from the lightning model class ###

from dentexmodel.models.toothmodel_basic import ToothModel

model = ToothModel(train_dataset=train_dataset,
                   batch_size=4,
                   num_workers=0)

dl = model.train_dataloader()
image_batch, label_batch = next(iter(dl))

print(image_batch.shape)
print(label_batch.shape)

# Plot one image from the batch with label
image_idx = 0
image = np.transpose(image_batch.numpy()[image_idx], axes=(1, 2, 0))
label = cl_dict.get(label_batch.numpy()[image_idx])
fig, ax = plt.subplots(figsize=(4,4))
ax.imshow(image, cmap='gray')
ax.set_title(label)
ax.set(xticks=[], yticks=[])
plt.show()
