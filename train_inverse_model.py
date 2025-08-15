###############################################################################
#
# Import functions
#
###############################################################################
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             '/home/rob/2D_ML/data_for_paper')))

import numpy as np
import random
import time
import gc

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model

from tensorflow.keras.layers import (Input, GRU, LayerNormalization,
                                     BatchNormalization, GroupNormalization,
                                     Dense, Attention, Dropout, LSTM,
                                     Bidirectional)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping)
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam

import NN_fns as NNf
import NN_variables as NNv

# seed for reproducibility
tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

# working directory
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + '/../')

###############################################################################
#
# Load arrays
#
###############################################################################
X_train_and_dev = np.load(os.path.join(dir_path, 'X_train_and_dev.npy'))
Y_train_and_dev = np.load(os.path.join(dir_path, 'Y_train_and_dev.npy'))

# shuffle in unison
X_train_and_dev, Y_train_and_dev = NNf.shuffle_arrays_in_unison(
    X_train_and_dev, Y_train_and_dev
)

# split: 85% train, 15% dev
size = X_train_and_dev.shape[0]
dev_size = int(size * 0.15)
train_size = size - dev_size

train_start = 0
train_end   = train_size
dev_start   = train_size
dev_end     = train_size + dev_size

X_train = X_train_and_dev[train_start:train_end]
X_dev   = X_train_and_dev[dev_start:dev_end]

Y_train = Y_train_and_dev[train_start:train_end]
Y_dev   = Y_train_and_dev[dev_start:dev_end]

# Z = concat(X, Y) for inverse training targets
Z_train = NNf.concat_X_and_Y(X_train, Y_train)
Z_dev   = NNf.concat_X_and_Y(X_dev,   Y_dev)

###############################################################################
#
# Prepare and re-initialize our models
#
###############################################################################
# build inverse (smaller) model
model_name_inverse = 'NN_inverse.keras'
model_inverse = NNv.build_model_inverse_smaller(
    NNv.n_points, NNv.num_IdVg, NNv.num_feats, NNv.num_params
)

###############################################################################
#
# Load saved forward model (do not retrain here)
#
###############################################################################
model_name_forward = 'NN_forward.keras'
forward_path = os.path.join(dir_path, 'models', model_name_forward)
model_forward = load_model(forward_path, compile=False)

###############################################################################
#
# Normal inverse, without pretraining
#
###############################################################################
model_inverse, _, = NNf.train_inverse_NN(
                                        X_train,
                                        Z_train,
                                        X_dev,
                                        Z_dev,
                                        model_inverse,
                                        model_name_inverse,
                                        model_forward,
                                        NNv.lr0_inverse,
                                        NNv.ar_inverse,
                                        NNv.N_anneals_inverse,
                                        NNv.patience_inverse,
                                        NNv.bs_inverse,
                                        )


save_dir = os.path.join(dir_path, 'models')
os.makedirs(save_dir, exist_ok=True)
inv_path = os.path.join(save_dir, model_name_inverse)
model_inverse.save(inv_path)
print('Saved inverse model to:', inv_path)
