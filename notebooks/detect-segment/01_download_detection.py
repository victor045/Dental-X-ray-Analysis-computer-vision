#!/usr/bin/env python
# coding: utf-8

# ## Dataset download for object detection ##

# In[1]:


# Imports
import os
import numpy as np
import pandas as pd
import tarfile
import random
import time
import glob
import json
from pathlib import Path
from dotenv import load_dotenv

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
from dentexmodel.dentexdata import DentexData
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData

print(f'Project module version: {dm.__version__}')

# ### Download the object detection data set ###
# Full data set: https://zenodo.org/records/7812323/files/training_data.zip?download=1
# 
# Object detection images: https://dsets.s3.amazonaws.com/dentex/dentex-quadrant-enumeration.tar.gz
# 

# In[2]:


# Path settings 
# Main data directory (defined as environment variable in docker-compose.yml)
data_root = os.environ.get('DATA_ROOT')

# Download directory (change as needed)
dentex_dir = os.path.join(data_root, 'dentex')
model_dir = os.path.join(data_root, 'model')
data_dir = os.path.join(dentex_dir, 'dentex_detection')
Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)

# This image directory is where the xrays are in the archive, so should be left as-is
image_dir = os.path.join(data_dir, 'quadrant_enumeration', 'xrays')

# Directory for the output
output_dir = os.path.join(data_dir, 'output')
Path(output_dir).mkdir(parents=True, exist_ok=True)

# In[3]:


# Create an instance of the DentexData class
dtx = DentexData(data_dir=data_dir)
url = dtx.detection_url
sz = FileOP().file_size_from_url(url)
sz_GB = sz/1.0e9

# Download and extract the data
print(f'Downloading {sz_GB:.2f} GB from:\n{url}')
data_tar_file = dtx.download_image_data(url=url)

# Check the images on disk
file_list = glob.glob(os.path.join(image_dir, '*.png'))
expected_n_images = 634
if not len(file_list) == expected_n_images:
    print(f'WARNING: expected number of images ({expected_n_images}) does not match the number of images on disk.')
    print(f'Delete files and start over.')
else:
    print(f'Extracted {len(file_list)} images.')

# In[4]:


# Create a data frame with the image file paths
file_name_list = [os.path.basename(file) for file in file_list]
im_number_list = [int(os.path.splitext(file)[0].rsplit('_', maxsplit=1)[-1]) for file in file_name_list]
files = pd.DataFrame({'image_number': im_number_list,
                      'file_name': file_name_list,
                      'file_path': file_list}).\
                sort_values(by='image_number', ascending=True).reset_index(drop=True)

display(files.head())

# In[5]:


# Load the annotation file
annotation_file = os.path.join(data_dir, 
                               'quadrant_enumeration', 
                               'train_quadrant_enumeration.json')
annotations = dtx.load_annotations(annotation_file)
print(f'Loaded annotations from file:\n{dtx.annotations_file}\n{annotations.keys()}')

# In[6]:


# Add image ids to the files data frame
js_im_df = pd.DataFrame(annotations.get('images')).\
                merge(files, on='file_name', how='inner').\
                sort_values(by='id', ascending=True).\
                reset_index(drop=True).\
                rename(columns={'id': 'image_id'}).\
                drop(['height', 'width'], axis=1)
display(js_im_df.head())
print(js_im_df.shape)

# In[7]:


# Create a dictionary of categories with IDs and names
# The categories are described in a dictionary at the top of the JSON file
display(dtx.annotations.keys())
print()
# We can create one dictionary with the labels for each annotation
# Annotations are quadrant (categories_1) and tooth position (categories_2).
categories = dtx.create_category_dict(categories=range(1, 3))
display(categories)

# In[8]:


# Save the data frame with the file paths and annotations
df_file_name = 'dentex_detection_dataset.parquet'
df_file = os.path.join(data_dir, df_file_name)
js_im_df.to_parquet(df_file)
print(f'Annotation data frame saved: {df_file}')
