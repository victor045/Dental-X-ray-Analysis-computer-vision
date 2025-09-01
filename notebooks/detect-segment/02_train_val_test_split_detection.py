#!/usr/bin/env python
# coding: utf-8

# ## Split data into training, validation and test sets ##

# In[1]:


# Imports
import os
import numpy as np
import pandas as pd

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import patches

# Appearance of the Notebook
from IPython.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
np.set_printoptions(linewidth=110)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# Import this module with autoreload
%load_ext autoreload
%autoreload 2
import dentexmodel as dm
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData
from dentexmodel.dentexdata import DentexData, val_test_split

print(f'Project module version: {dm.__version__}')

# In[2]:


# Path settings 
# Main data directory (defined as environment variable in docker-compose.yml)
data_root = os.environ.get('DATA_ROOT')

# Download directory (change as needed)
dentex_dir = os.path.join(data_root, 'dentex')
model_dir = os.path.join(data_root, 'model')
data_dir = os.path.join(dentex_dir, 'dentex_detection')

# This image directory is where the xrays are in the archive, so should be left as-is
image_dir = os.path.join(data_dir, 'quadrant_enumeration', 'xrays')

# Directory for the output
output_dir = os.path.join(data_dir, 'output')

# Data frame with images and paths
data_df_file_name = 'dentex_detection_dataset.parquet'
data_df_file = os.path.join(data_dir, data_df_file_name)

# In[3]:


# Load the data frame with image paths and bounding boxes
data_df = pd.read_parquet(data_df_file)
display(data_df.head(2))
# Create an instance of the DentexData class
dtx = DentexData(data_dir=data_dir)

# In[4]:


# Function to create the data splits
label_col = 'label'
dset_df = val_test_split(data=data_df, 
                         n_test_per_class=50,
                         n_val_per_class=50)

# In[5]:


train_set = set(dset_df.loc[dset_df['dataset'] == 'train', 'file_name'].values)
print(f'We have {len(train_set)} images in the train set.')

val_set = set(dset_df.loc[dset_df['dataset'] == 'val', 'file_name'].values)
print(f'We have {len(val_set)} images in the validation set.')

test_set = set(dset_df.loc[dset_df['dataset'] == 'test', 'file_name'].values)
print(f'We have {len(test_set)} images in the test set.')

# Make sure that these data sets are distinct
print(train_set.intersection(val_set))
print(train_set.intersection(test_set))
print(val_set.intersection(test_set))

# In[6]:


# Save the data split
datasplit_file_name = 'dentex_detection_datasplit.parquet'
datasplit_file = os.path.join(data_dir, datasplit_file_name)
dset_df.to_parquet(datasplit_file)
print(datasplit_file)
display(dset_df.head())
