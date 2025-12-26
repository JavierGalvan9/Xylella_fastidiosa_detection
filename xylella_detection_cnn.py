# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 10:59:22 2022

@author: UX325
"""

import argparse
import csv
import glob
import os
import random
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

from callbacks import MyCallback
from data_preprocessing import data_preprocessing
from models import ResNet50_model, VGG16_model, inceptionV3_model

parentDir = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(parentDir, "general_utils"))
import file_management

seed(0)

print(tf.config.list_physical_devices('GPU'))
# Use the first available GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only use the memory needed
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
# tf.config.threading.set_intra_op_parallelism_threads(1)
# tf.config.threading.set_inter_op_parallelism_threads(1)

parser = argparse.ArgumentParser()
parser.add_argument("--num_epochs", 
                    type=int,
                    default=2,
                    help="especify directory to save data")

parser.add_argument("--oversampling", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-oversampling", dest="oversampling", action="store_false")

parser.add_argument("--undersampling", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-undersampling", dest="undersampling", action="store_false")

parser.add_argument("--cost_sensitive", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-cost_sensitive", dest="cost_sensitive", action="store_false")

parser.add_argument("--dropout", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-dropout", dest="dropout", action="store_false")

parser.add_argument("--use_spectral_bands", 
                    action="store_true",
                    default=True,
                    help="Choose whether to use only spectral bands or not")
parser.add_argument("--no-use_spectral_bands", dest="use_spectral_bands", action="store_false")

parser.add_argument("--use_indices", 
                    action="store_true",
                    default=True,
                    help="Choose whether to use only indices or not")
parser.add_argument("--no-use_indices", dest="use_indices", action="store_false")

parser.add_argument("--L2_regularizer", 
                    action="store_true",
                    default=False,
                    help="especify directory to save data")
parser.add_argument("--no-L2_regularizer", dest="L2_regularizer", action="store_false")

args = parser.parse_args()

## Create dir to save classification results and filename based on arguments
classification_main_path = 'Classification results'
args_path = ''
for arg, value in vars(args).items():
    if value:
        args_path += f'{arg}_{value} '

# If there are no arguments, use the default name
if len(args_path) > 0:
    classification_path = os.path.join(classification_main_path, args_path)
else:
    classification_path = os.path.join(classification_main_path, 'Baseline')

# If the directory already exists, add a number to the end of the name
fileCounter = len(glob.glob1(classification_main_path,args_path+'*'))
if fileCounter > 0:
    classification_path += str(fileCounter)
os.makedirs(classification_path, exist_ok=True)

# Load the training and test sets
df = file_management.load_lzma('Processed Data/QPCR_labelled_df.lzma')
df.loc[df['PCR'] ==np.nan, 'PCR'] = 0
spectral_bands = ['C', 'B', 'G', 'Y', 'R', 'RE', 'N', 'N2']
X = df.loc[:, spectral_bands + list(df.columns[8:-4])].values
Y = df['PCR'].values
print('X shape:', X.shape)
# Number of features
n_features = X.shape[1]
print('Number of features:', n_features)
pan_shape = (808, 967)
X = X.reshape(pan_shape[0], pan_shape[1], n_features)
# Only take the 3 first features from X
X = X[:, :, :3]

print('X shape:', X.shape)
# Convert input images to RGB format by replicating the same data across all 3 channels
# X = np.repeat(X[..., np.newaxis], 3, axis=-1)
# print('X shape:', X.shape)

# X = np.reshape(X, (X.shape[2], X.shape[0], X.shape[1], X.shape[3]))
# print('X shape:', X.shape)

## Train simple NN with 7-fold cross-validation
# Choose the model architecture based on the arguments passed to the script 
classifier = ResNet50_model(input_shape=X.shape)

#Set the training parameters 
num_epochs = args.num_epochs
verbosity = 0 #0:silent, 1:to show a progress bar during the training, 2:show results after each epoch
batch_size = 1

print(X.shape)

# Train the model
classifier.fit(X, Y, epochs=10, batch_size=batch_size)
scores = classifier.evaluate(X, Y, verbose=0)

print(scores)
