#!/usr/bin/env python
# coding: utf-8

# ## Create data sets from bounding boxes ##
# In this notebook, single-tooth images are created from the panoramic xrays

# Imports
import os
import numpy as np
import pandas as pd
import tarfile
import random
import time
import glob
import json
import cv2
from pathlib import Path

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import patches

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData, crop_image

print(f'Project module version: {dm.__version__}')

# Main data directory (defined as environment variable in docker-compose.yml)
data_root = os.environ.get('DATA_ROOT')

# Download directory (change as needed)
dentex_dir = os.path.join(data_root, 'dentex')
model_dir = os.path.join(data_root, 'model')
data_dir = os.path.join(dentex_dir, 'dentex_classification')

# This image directory is where the xrays are in the archive, so should be left as-is
image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays')

# Directory for the output
output_dir = os.path.join(data_dir, 'output')

df_file_name = 'dentex_disease_dataset.parquet'
df_file = os.path.join(data_dir, df_file_name)

# Load the annotation data frame
an_df = pd.read_parquet(df_file)
print()
print(f"{content}")
# ### Create the cropped images from the bounding boxes ###

# File path for the cropped images
cropped_image_dir = os.path.join(image_dir, 'crop')
Path(cropped_image_dir).mkdir(exist_ok=True)
file_name_list = sorted(an_df['file_name'].unique())

# Start a list of new data frames
data_df_list = []

# Loop over the panoramic x-rays
for f, file_name in enumerate(file_name_list):
    box_name_list = an_df.loc[an_df['file_name'] == file_name, 'box_name'].values
    if (f + 1) % 50 == 0:
        print(f'Processing image {f+1} / {len(file_name_list)}')
    # Loop over the bounding boxes for this file
    for b, box_name in enumerate(box_name_list):
        box_file = os.path.join(cropped_image_dir, f'{box_name}.png')
        
        # Get the row in the data frame
        box_df = an_df.loc[(an_df['file_name'] == file_name) & (an_df['box_name'] == box_name)].\
                        assign(box_file=box_file)
        box = box_df['bbox'].values[0]
        bbox = box[0], box[1], box[0] + box[2], box[1] + box[3]
        label = box_df['label'].values[0]
        file = os.path.join(image_dir, file_name)

        if not os.path.exists(box_file):
            
            # Load the image and then crop it
            im = ImageData().load_image(file)
            im_crop = crop_image(im, bbox)
                
            # Some contrast enhancement
            im_crop_enhanced = ImageData().hist_eq(im_crop)
            
            # Save the image
            cv2.imwrite(box_file, cv2.cvtColor(im_crop_enhanced, cv2.COLOR_RGB2BGR))

        # Add the image size to the data frame
        box_file_size = ImageData().image_size(box_file)
        box_df = box_df.assign(im_width=box_file_size[1],
                               im_height=box_file_size[0])
        
        # Add the data frame for this image to the list
        data_df_list.append(box_df)

# Concatenate the data frames
data_df = pd.concat(data_df_list, axis=0, ignore_index=True)

# Save the data frame
print(f"{content}")
df_box_file = df_file.replace('dataset', 'cropped_dataset')
print(df_box_file)
data_df.to_parquet(df_box_file)
