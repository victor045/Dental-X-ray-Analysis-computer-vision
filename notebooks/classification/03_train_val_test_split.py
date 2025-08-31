#!/usr/bin/env python
# coding: utf-8

# ## Split data into training, validation and test sets ##

# Imports
import os
import numpy as np
import pandas as pd

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import patches

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData
from dentexmodel.dentexdata import val_test_split

print(f'Project module version: {dm.__version__}')

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

df_box_file_name = 'dentex_disease_cropped_dataset.parquet'
df_box_file = os.path.join(data_dir, df_box_file_name)

# Load the data frame with image paths and bounding boxes
data_df = pd.read_parquet(df_box_file)
print(f"{content}")
# Function to create the data splits
label_col = 'label'
dset_df = val_test_split(data=data_df, 
                         label_col=label_col,
                         n_test_per_class=30,
                         n_val_per_class=30)

# Make sure that we have three non-overlapping data sets
train_set = set(dset_df.loc[dset_df['dataset']=='train', 'box_name'].values)
print(f'We have {len(train_set)} images in the train set.')

val_set = set(dset_df.loc[dset_df['dataset']=='val', 'box_name'].values)
print(f'We have {len(val_set)} images in the validation set.')

test_set = set(dset_df.loc[dset_df['dataset']=='test', 'box_name'].values)
print(f'We have {len(test_set)} images in the test set.')
print()

# Make sure that these data sets are distinct
print(train_set.intersection(val_set))
print(train_set.intersection(test_set))
print(val_set.intersection(test_set))

# Save the data split
datasplit_file_name = 'dentex_disease_datasplit.parquet'
datasplit_file = os.path.join(data_dir, datasplit_file_name)
dset_df.to_parquet(datasplit_file)
print(datasplit_file)
print(f"{content}")
