#!/usr/bin/env python
# coding: utf-8

# ## Performance Metrics and Logging ##
# Logging of performance metrics during training an evaluation

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
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
import torchmetrics

# Albumentations library
import albumentations as alb

# Appearance of the Notebook
# Import this module with autoreload
import dentexmodel as dm
from dentexmodel.fileutils import FileOP
from dentexmodel.imageproc import ImageData
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

# The output of this transformation must match the required input size for the model
max_image_size = 550
im_size = 224

# For validation and testing, we do not want any augmentations
# but we will still need the correct input size and image normalization
val_transform = alb.Compose([
    alb.Resize(im_size, im_size),
    alb.Normalize(mean=ImageData().image_net_mean, 
                  std=ImageData().image_net_std)])

# Create the test data set from the data frame
test_dataset = DatasetFromDF(data=test_df,
                             file_col='box_file',
                             label_col='cl',
                             max_image_size=max_image_size,
                             transform=val_transform,
                             validate=True)

# ### Load model from checkpoint ###

from dentexmodel.models.toothmodel_fancy import ToothModel
link = 'https://dsets.s3.amazonaws.com/dentex/toothmodel_fancy_40.ckpt'

# Let's see if we have a saved checkpoint
# The previous training notebook should have the defined model name
model_name = 'FancyLR'
model_version = 1

# The latest checkpoint should be here
checkpoint_dir = os.path.join(model_dir, 
                              model_name,
                              f'version_{model_version}',
                              'checkpoints')

last_checkpoint = glob.glob(os.path.join(checkpoint_dir, 'last.ckpt'))
if len(last_checkpoint) > 0:
    print(f'Using checkpoint file "last.ckpt" in {checkpoint_dir}.')
    checkpoint_file = last_checkpoint[0]
else:
    print(f'Last checkpoint file "last.ckpt" not found in {checkpoint_dir}.')
    print(f'Downloading checkpoint from {link}')
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_file = FileOP().download_from_url(url=link, download_dir=checkpoint_dir)

batch_size = 16
num_classes = 4
model = ToothModel.load_from_checkpoint(checkpoint_file,
                                        test_dataset=test_dataset, 
                                        map_location=device,
                                        batch_size=batch_size,
                                        num_classes=num_classes,
                                        num_workers=1)

# We try the metrics with the trained model
# Load a test batch
dl = model.test_dataloader()
test_image_batch, test_label_batch = next(iter(dl))
print(f"{content}")
print(f"{content}")
# Forward - pass on the test batch
pred = model(test_image_batch.to(device))

# ### Performance metrics: sklearn.metrics library ###
# Here is a good description of performance metrics for multi-class classification
# https://www.evidentlyai.com/classification-metrics/multi-class-metrics

# True and predicted class labels
true_cl = test_label_batch.numpy()
pred_cl = torch.argmax(pred, dim=1).detach().cpu().numpy()
# From seeing this, the accuracy should be as shown
print(f'True class labels:      {true_cl}')
print(f'Predicted class labels: {pred_cl}')

# Accuracy
from sklearn.metrics import accuracy_score
acc = sum(pred_cl == true_cl)/batch_size
sk_acc = accuracy_score(y_true=true_cl, y_pred=pred_cl)
print(f'The accuracy should be: {acc:.3f}')
print(f'Scikit-Learn:           {sk_acc}')

# Precision for binary classification
# Precision is a metric that quantifies the number of correct predictions
# TruePositives / (TruePositives + FalsePositives)
prec_list = []
for cl in range(num_classes):
    true_bin_cl = [1 if label==cl else 0 for label in true_cl]
    pred_bin_cl = [1 if label==cl else 0 for label in pred_cl]
    true_positives_list = [1 if true_bin_cl[i]==pred_bin_cl[i]==1 else 0 for i in range(batch_size)]
    false_positives_list = [1 if (true_bin_cl[i]==0 and pred_bin_cl[i]==1) else 0 for i in range(batch_size)]
    true_positives = sum(true_positives_list)
    false_positives = sum(false_positives_list)
    
    # If there are no positive predictions in sample, we cannot calculate precision
    if true_positives==false_positives==0:
        prec_cl = 0
    else:
        prec_cl = true_positives / (true_positives + false_positives)
    print()
    print(f'Precision for class label {cl}: {prec_cl}')
    if prec_cl==0:
        print(f'WARNING: No positive samples in batch for label {cl}.')
    print(true_bin_cl)
    print(pred_bin_cl)
    prec_list.append(prec_cl)

# Calculate average precision across classes
print(f'Average precision: {sum(prec_list)/num_classes: .3f}')

# Precision from Scikit-Learn library
from sklearn.metrics import precision_score
sk_prec = precision_score(y_true=true_cl, y_pred=pred_cl, average='macro')
print(f'Scikit-Learn:      {sk_prec: .3f}')

# Show the calculations from Scikit-Learn
print(f'Accuracy:  {sk_acc}')
print(f'Precision:{sk_prec: .3f}')
# Recall
from sklearn.metrics import recall_score
sk_rec = recall_score(y_true=true_cl, y_pred=pred_cl, average='macro')
print(f'Recall:   {sk_rec: .3f}')

# F1 score
from sklearn.metrics import f1_score
sk_f1 = f1_score(y_true=true_cl, y_pred=pred_cl, average='macro')
print(f'F1:       {sk_f1: .3f}')

# AUC score
from sklearn.metrics import roc_auc_score
# For this, we need probability estimates for each class
sm = nn.Softmax(dim=1)
pred_cl_score = sm(pred).detach().cpu().numpy()
sk_auc = roc_auc_score(y_true=true_cl, y_score=pred_cl_score, average='macro', multi_class='ovr')
print(f'AUC:      {sk_auc: .3f}')

# ### Performance metrics: torchmetrics.classification library ###

true_cl = test_label_batch.to(device)
print(f"{content}")
# Probability estimates for each class
sm = nn.Softmax(dim=1)
pred_cl_score = sm(pred)
pred_cl = torch.argmax(pred_cl_score, dim=1)
print(f"{content}")
print()

# Accuracy
from torchmetrics.classification import MulticlassAccuracy
acc = MulticlassAccuracy(num_classes=4, average='micro').to(device)
tm_acc = acc(preds=pred_cl_score, target=true_cl)
print(f'Accuracy:  {tm_acc}')

# Precision
from torchmetrics.classification import MulticlassPrecision
prec = MulticlassPrecision(num_classes=4, average='macro').to(device)
tm_prec = prec(preds=pred_cl_score, target=true_cl)
print(f'Precision:{tm_prec: .3f}')

# Recall
from torchmetrics.classification import MulticlassRecall
rec = MulticlassRecall(num_classes=4, average='macro').to(device)
tm_rec = rec(preds=pred_cl_score, target=true_cl)
print(f'Recall:   {tm_rec: .3f}')  

# F1
from torchmetrics.classification import MulticlassF1Score
f1 = MulticlassF1Score(num_classes=4, average='macro').to(device)
tm_f1 = f1(preds=pred_cl_score, target=true_cl)
print(f'F1:       {tm_f1: .3f}')

# AUC
from torchmetrics.classification import MulticlassAUROC
auc = MulticlassAUROC(num_classes=4, average='macro').to(device)
tm_auc = auc(preds=pred_cl_score, target=true_cl)
print(f'AUC:      {tm_auc: .3f}')

test_dict = nn.ModuleDict()

metric_dict = nn.ModuleDict({'acc': MulticlassAccuracy(num_classes=4, average='micro').to(device),
                             'prec': MulticlassPrecision(num_classes=4, average='macro').to(device)})

metric_prefix = 'train'
preds = sm(pred)
target = true_cl

performance_dict_1 = {}
for metric_name, metric in metric_dict.items():
    performance_dict_1.update({f'{metric_prefix}_{metric_name}': metric(preds=preds, target=target)})

performance_dict_2 = {}
for metric_name, metric in metric_dict.items():
    performance_dict_2.update({f'{metric_prefix}_{metric_name}': metric(preds=preds, target=target)+0.1})

performance_list = [performance_dict_1, performance_dict_2]
print(f"{content}")
# Now, we have to average the tensors in the dictionaries
key_list = list(performance_list[0].keys())
print(key_list)
key = key_list[0]
print(key)

epoch_performance_dict = {}
for metric_name in key_list:
    m = torch.stack([x.get(metric_name) for x in performance_list])
    n = m.mean().detach().cpu().numpy().round(2)
    epoch_performance_dict.update({metric_name: n})

print(epoch_performance_dict)
