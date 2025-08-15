###############################################################################
#
# Import functions
#
###############################################################################
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/rob/2D_ML/data_for_paper')))

import copy
import matplotlib as mpl
import sys
import glob as glob
import ast
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy.stats
from sklearn.utils import shuffle
import time

import tensorflow as tf
import gc
from tensorflow.keras.layers import Input, GRU, LayerNormalization, BatchNormalization, GroupNormalization, Dense, Attention, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers  import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import NN_fns as NNf
import NN_variables as NNv

# seed for reproducibility
tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

# working directory
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + '/../')

X_train_and_dev = np.load(dir_path + '/X_train_and_dev.npy')
Y_train_and_dev = np.load(dir_path + '/Y_train_and_dev.npy')

#######################################################################
#
# Draw our bootstrapped training and dev sets
#
#######################################################################

X_train_and_dev, Y_train_and_dev = NNf.shuffle_arrays_in_unison(
                                  X_train_and_dev, Y_train_and_dev)

# uncomment this line to train with all data
size = np.shape(X_train_and_dev)[0]
dev_size = int(size*0.15)
train_size = size - dev_size

train_start = 0
train_end = train_size
dev_start = train_size
dev_end = train_size + dev_size

X_train = X_train_and_dev[train_start:train_end:]
X_dev = X_train_and_dev[dev_start:dev_end:]
X_test = np.load(dir_path + '/X_test.npy')

Y_train = Y_train_and_dev[train_start:train_end]
Y_dev = Y_train_and_dev[dev_start:dev_end]
Y_test = np.load(dir_path + '/Y_test.npy')

Z_train = NNf.concat_X_and_Y(X_train, Y_train)
Z_dev = NNf.concat_X_and_Y(X_dev, Y_dev)

#######################################################################
#
# Prepare and re-initalize our models
#
#######################################################################

model_name_forward = 'NN_forward.keras'
model_forward = NNv.build_model_forward(NNv.num_params, NNv.num_IdVg)

#######################################################################
#
# forward only
#
#######################################################################
model_forward, _ = NNf.train_forward_NN(
    X_train[:, :, 0:NNv.num_IdVg*2],
    Y_train,
    X_dev[:, :, 0:NNv.num_IdVg*2],
    Y_dev,
    model_forward,
    model_name_forward,
    NNv.lr0_forward,
    NNv.ar_forward,
    NNv.N_anneals_forward,
    NNv.patience_forward,
    NNv.bs_forward
)

save_dir = os.path.join(dir_path, "models")
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, model_name_forward)  # '.../models/NN_forward.keras'
model_forward.save(save_path)
print(f"Saved forward model to: {save_path}")
