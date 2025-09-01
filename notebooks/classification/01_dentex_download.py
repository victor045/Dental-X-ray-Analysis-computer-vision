#!/usr/bin/env python
# coding: utf-8
## Dataset download ##

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

# Matplotlib for plotting
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from matplotlib import patches

# Import this module
import dentexmodel as dm
from dentexmodel.dentexdata import DentexData
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData

print(f'Project module version: {dm.__version__}')

### Download the classification data set ###
# Full data set: https://zenodo.org/records/7812323/files/training_data.zip?download=1

# Main data directory (defined as environment variable in docker-compose.yml)
data_root = os.environ.get('DATA_ROOT')

# Download directory (change as needed)
dentex_dir = os.path.join(data_root, 'dentex')
model_dir = os.path.join(data_root, 'model')
data_dir = os.path.join(dentex_dir, 'dentex_classification')

Path(data_dir).mkdir(parents=True, exist_ok=True)
Path(model_dir).mkdir(parents=True, exist_ok=True)

# This image directory is where the xrays are in the archive, so should be left as-is
image_dir = os.path.join(data_dir, 'quadrant-enumeration-disease', 'xrays')

# Lets create a directory for the output
output_dir = os.path.join(data_dir, 'output')
Path(output_dir).mkdir(exist_ok=True, parents=True)

# Create an instance of the DentexData class
dtx = DentexData(data_dir=data_dir)
url = dtx.classification_url
sz = FileOP().file_size_from_url(url)
sz_GB = sz/1.0e9

# Download and extract the data
print(f'Downloading {sz_GB:.2f} GB from:\n{dtx.classification_url}')
data_tar_file = dtx.download_image_data(url=url)

# Check the images on disk
file_list = glob.glob(os.path.join(image_dir, '*.png'))
expected_n_images = 705
if not len(file_list) == expected_n_images:
    print(f'WARNING: expected number of images ({expected_n_images}) does not match the number of images on disk.')
    print(f'Delete files and start over.')
else:
    print(f'Extracted {len(file_list)} images.')

# Create a data frame with the image file paths
file_name_list = [os.path.basename(file) for file in file_list]
im_number_list = [int(os.path.splitext(file)[0].rsplit('_', maxsplit=1)[-1]) for file in file_name_list]
files = pd.DataFrame({'image_number': im_number_list,
                      'file_name': file_name_list,
                      'file_path': file_list}).\
                sort_values(by='image_number', ascending=True).reset_index(drop=True)

print("Files DataFrame:")
print(files.head())

# Load the annotation file
annotation_file = os.path.join(data_dir, 
                               'quadrant-enumeration-disease', 
                               'train_quadrant_enumeration_disease.json')
annotations = dtx.load_annotations(annotation_file)
print(f'Loaded annotations from file:\n{dtx.annotations_file}\n{annotations.keys()}')

# Add image ids to the files data frame
js_im_df = pd.DataFrame(annotations.get('images')).\
                merge(files, on='file_name', how='inner').\
                sort_values(by='id', ascending=True).\
                reset_index(drop=True).\
                rename(columns={'id': 'image_id'}).\
                drop(['height', 'width'], axis=1)
print("Merged DataFrame:")
print(js_im_df.head())
print(js_im_df.shape)

### Disease categories ###

# Create a dictionary of categories with IDs and names
# The categories are described in a dictionary at the top of the JSON file
print("Annotation keys:")
print(dtx.annotations.keys())
print()
# We can create one dictionary with the labels for each annotation
categories = dtx.create_category_dict()
print("Categories:")
print(categories)

### Transfer annotations from json file into a data frame ###
# The data frame format is needed to create the classification data set

# Loop over the annotations
an_df_list = []
for idx, an_dict in enumerate(annotations.get('annotations')):
    if (idx + 1) % 500 == 0:
        print(f'Annotation {idx + 1} / {len(annotations.get("annotations"))}')
    
    image_id = an_dict.get('image_id')
    id_df = js_im_df.loc[js_im_df['image_id'] == image_id]
    
    # Find the quadrant, tooth position and disease categories for this annotation
    quadrant_id = an_dict.get('category_id_1')
    quadrant = categories.get('categories_1').get(quadrant_id)
    
    position_id = an_dict.get('category_id_2')
    position = categories.get('categories_2').get(position_id)
    
    disease_id = an_dict.get('category_id_3')
    disease = categories.get('categories_3').get(disease_id)
    
    id_df = id_df.assign(quadrant=quadrant,
                         position=position,
                         label=disease,
                         cl = disease_id,
                         area=[an_dict.get('area')],
                         bbox=[an_dict.get('bbox')],
                         box_name=(f'{os.path.splitext(id_df["file_name"].values[0])[0]}_'
                                   f'{idx}_{quadrant}_{position}'))    
    an_df_list.append(id_df)
an_df = pd.concat(an_df_list, axis=0, ignore_index=True)

# Add the number of annotations to each image
n_annotations = an_df[['file_name', 'label']].\
                groupby('file_name').count().\
                reset_index(drop=False).\
                rename(columns={'label': 'annotations'})

an_df = an_df.merge(n_annotations, on='file_name', how='inner').\
                sort_values(by='image_id', ascending=True).\
                reset_index(drop=True)

print()
print("Final annotations DataFrame:")
print(an_df.head())

### Plot some images with annotations ###

# Show an xray image with bounding boxes
np.random.seed(123)
file_name_list = np.random.choice(an_df['file_name'].unique(), size=5, replace=False)  # Reduced to 5 for speed
#file_name_list = ['train_265.png', 'train_269.png', 'train_270.png']
for file_name in file_name_list:
    an_file_df = an_df.loc[(an_df['file_name'] == file_name)] 
    #                        (an_df['position'] == '6') &
    #                        (an_df['quadrant'] == '4')]
    
    file = os.path.join(image_dir, file_name)
    im = ImageData().load_image(file)
    #im = ImageData().hist_eq(im)
    
    # Create a list of colors for the rectangles
    color = cm.rainbow(np.linspace(0, 1, len(an_file_df)))
    color_list = [color[c] for c in range(len(color))]
    text_offset_xy = (-10, -10)
    
    # Create the figure and show the panoramic x-ray image
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.imshow(im)
    
    # Loop over the bounding boxes
    for i, idx in enumerate(an_file_df.index): 
    
        box_df = an_file_df.loc[an_file_df.index==idx]
        
        box = box_df['bbox'].values[0]
        bbox = box[0], box[1], box[0] + box[2], box[1] + box[3]
        label = box_df['label'].values[0]
        
        
        rect = patches.Rectangle(xy=(box[0], box[1]), width=box[2], height=box[3], 
                                         linewidth=1.5, edgecolor=color_list[i], 
                                         facecolor='none', alpha=0.7)
        ax.add_patch(rect)
        
        ax.text(box[0]+text_offset_xy[0], 
                box[1]+text_offset_xy[1]+i, 
                label, 
                color='r',
                fontsize='small',
                fontweight='medium')
        
        ax.set(xticks=[], yticks=[])


    image_name = f'{os.path.splitext(file_name)[0]}_boxes.png'
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight')
    plt.close()  # Close the figure to save memory

# Save the data frame with the file paths and annotations
df_file_name = 'dentex_disease_dataset.parquet'
df_file = os.path.join(data_dir, df_file_name)
an_df.to_parquet(df_file)
print(f'Annotation data frame saved: {df_file}')

