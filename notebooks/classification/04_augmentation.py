#!/usr/bin/env python
# coding: utf-8

# ## Preprocessing and data augmentations ##

# Imports
import os
import numpy as np
import pandas as pd
import cv2

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm

# PyTorch
import torch

# Albumentations library
import albumentations as alb

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.imageproc import ImageData
from dentexmodel.torchdataset import load_and_process_image

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

# Load the data frame with image paths
data_df = pd.read_parquet(data_file)
print(f"{content}")
# ### Preprocessing of the images ###

# All of the images have different sizes. For the model we need all images to be the same shape. It is not always a good idea to change the aspect ratio of the images and we also want to keep the size of the teeth relative to the whole image size.
# This is the process:
# 1. Find the largest image dimension and size S for that dimension in the entire data set
# 2. Scale all smaller images (with original aspect ration maintained) so that their largest dimension is S
# 3. Pad the smaller dimension to get square images
# 4. Scale the square image to the input size of the model

# We have the images sizes in the data frame
# Finding the largest image and its dimensions is easy
im_max_width = data_df['im_width'].max()
im_max_height = data_df['im_height'].max()
print(f'Maximum image width across the data set:  {im_max_width}')
print(f'Maximum image height across the data set: {im_max_height}')
# Set the maximum image size for all images
im_max_size = int(np.max((im_max_width, im_max_height)))
print(f'Maximum image size: {im_max_size}')

# Take a look at a few images
np.random.seed(234)
image_list = np.random.choice(data_df['box_name'].unique(), size=5, replace=False)
for i, image in enumerate(image_list):
    # Load the image
    image_file = os.path.join(cropped_image_dir, f'{image}.png')
    im = ImageData().load_image(image_file)
    # Pre-process the image
    im_processed = load_and_process_image(image_file_path=image_file,
                                          max_image_size=im_max_size)
    # Show the two images side-by-side
    im_list = [im, im_processed]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(6, 2))
    for a, axa in enumerate(ax):
        axa.imshow(im_list[a])
        axa.set_title(f'Image shape {im_list[a].shape}')
        axa.set(xticks=[], yticks=[])
    plt.show()

# ### Run some example augmentations ###

# This is the input size for the model
im_size = 512
# Here is a list of example augmentations
# A complete list of pixel-level transformations is here
# https://github.com/albumentations-team/albumentations#pixel-level-transforms
train_aug = alb.Compose([
    alb.Resize(im_size + 32, im_size + 32),
    alb.RandomCrop(im_size, im_size),
    alb.HorizontalFlip(),
    alb.ShiftScaleRotate(),
    alb.Blur(),
    alb.RandomGamma(),
    alb.Sharpen(),
    alb.GaussNoise(),
    alb.CoarseDropout(16, 32, 32),
    alb.CLAHE()])
# Set the number of random transformations
n_transforms = 15

figsize = (15, 15)
for image in image_list:
    image_file = os.path.join(cropped_image_dir, f'{image}.png')
    im = load_and_process_image(image_file_path=image_file, max_image_size=im_max_size)
    fig, ax = plt.subplots(nrows=1, ncols=n_transforms, figsize=figsize)
    for n in range(n_transforms):
        im_aug = train_aug(image=im)['image']
        ax[n].imshow(im_aug)
        ax[n].set(xticks=[], yticks=[])
    plt.show()
