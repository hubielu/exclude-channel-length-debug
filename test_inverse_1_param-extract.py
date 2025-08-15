###############################################################################
#
# Import functions
#
###############################################################################

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
import matplotlib.font_manager as font_manager
import tensorflow as tf
from tensorflow.keras.layers import Input, GRU, LayerNormalization, BatchNormalization, Dense, Attention, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

import NN_fns as NNf
import NN_variables as NNv

font_dir = ['/home/rob/Lato']
for font in font_manager.findSystemFonts(font_dir):
    font_manager.fontManager.addfont(font)
mpl.rcParams['font.family'] = 'Lato'

mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markeredgewidth'] = 1
mpl.rcParams['axes.linewidth'] = 0.7
mpl.rcParams['xtick.major.width'] = 0.4
mpl.rcParams['ytick.major.width'] = 0.4
mpl.rcParams['xtick.minor.width'] = 0.2
mpl.rcParams['ytick.minor.width'] = 0.2
plt.rcParams['xtick.minor.visible'] = True
plt.rcParams['ytick.minor.visible'] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['lines.markersize'] = 5

fontsize = 9
mpl.rcParams.update({'font.size': fontsize})

# seed for reproducibility
tf.random.set_seed(19700101)
np.random.seed(19700101)
random.seed(19700101)

blue = '#19546d'
red = '#bd2b49'
purple = '#192a6d'

# working directory
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + '/../')

model_inverse = load_model(
    os.path.join(dir_path, 'NN_inverse.keras'),
    custom_objects={'surrogate_loss': NNv.dummy_fn},
    compile=False
)

X_test = np.load(os.path.join(dir_path, 'X_test.npy'))
Y_test = np.load(os.path.join(dir_path, 'Y_test.npy'))

Xscaling = np.loadtxt(os.path.join(dir_path, 'Xscaling.dat'))
Xmins = Xscaling[0, :]
Xmaxs = Xscaling[1, :]

Yscaling = np.loadtxt(os.path.join(dir_path, 'Yscaling.dat'))
Ymins = Yscaling[0, :]
Ymaxs = Yscaling[1, :]

Y_pred = np.array(model_inverse.predict(X_test))[:, 0:NNv.num_params]

ticks = [
    [0, 50, 100, 150, 200],                 # Channel length
    [0, 15, 30],
    [0, 125, 250, 375, 500],
    [0, 0.5, 1],
    [0, 1, 2, 3],
    [0, 50, 100, 150, 200],
    [0, 50, 100, 150, 200],
    [0, 1, 2, 3],
    [50, 175, 300],
]

variables = [
    'Channel length (nm)',
    'Mobility (cm$^2$  V$^{-1}$ s$^{-1}$)',
    'Schottky barrier height (meV)',
    'Effective density of states ($\\times 10^{13}$ cm$^{-2}$)',
    'Peak donor density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
    'Donor energy mid (meV below conduction band edge)',
    'Donor energy width (meV)',
    'Peak acceptor band tail density ($\\times$ 10$^{13}$ cm$^{-2}$ eV$^{-1}$)',
    'Acceptor band tail energy width (meV)',
]

# Dynamic subset: up to 250 or dataset size, whichever is smaller
N = Y_test.shape[0]
subset_idx = np.arange(min(250, N))


for j in range(9):
    fig, axs = plt.subplots(1, 2, figsize=(3.5, 2.25))
    plt.subplots_adjust(left=0.13, top=0.71, right=0.9, bottom=0.175, hspace=0.5, wspace=0.7)

    Ymin = float(Ymins[j])
    Ymax = float(Ymaxs[j])

    if j in [3]:
        Ymin *= 6.15e-8 / 1e13
        Ymax *= 6.15e-8 / 1e13
    elif j in [4]:
        Ymin /= 1e13
        Ymax /= 1e13
    if j in [7]:
        Ymin *= 6.15e-8 / 1e13
        Ymax *= 6.15e-8 / 1e13
    elif j in [5, 6, 8]:
        Ymin *= 1000
        Ymax *= 1000

    y_true = NNf.unscale_vector(Y_test[:, j], Ymin, Ymax)
    y_pred = NNf.unscale_vector(Y_pred[:, j], Ymin, Ymax)

    if j == 2:
        y_true = 5000 - 1000 * y_true
        y_pred = 5000 - 1000 * y_pred
        Ymin = 0.0
        Ymax = 500.0

    axs[0].plot(
        y_true[subset_idx],
        y_pred[subset_idx],
        marker='o',
        ls='None',
        markersize=4,
        color='k',
        markerfacecolor=purple,
        markeredgewidth=0.4
    )

    axs[0].plot(
        [-10000, 10000],
        [-10000, 10000],
        color=red,
        ls='--'
    )

    axs[0].set_xlim([Ymin, Ymax])
    axs[0].set_ylim([Ymin, Ymax])
    if j < len(ticks):
        axs[0].set_xticks(ticks[j])
        axs[0].set_yticks(ticks[j])

    errors = (y_true - y_pred)
    MAE = float(np.median(np.abs(errors)))
    std = float(np.std(errors))

    std_safe = max(std, 1e-12)
    binmin = -4.0 * std_safe
    binmax =  4.0 * std_safe
    bins = np.linspace(binmin, binmax, 26)

    axs[1].hist(
        errors,
        bins=bins,
        color=purple,
        edgecolor='k',
        linewidth=0.15
    )

    axs[0].set_xlabel('Actual')
    axs[0].set_ylabel('Predicted')
    axs[1].set_xlabel('Error')
    axs[1].set_ylabel('Counts')

    fig.text(0.5, 0.88, variables[j], ha='center', fontsize=10.5)
    fig.suptitle('\n \n Median absolute error = {} \n Standard deviation of error = {}'.format(
        round(MAE, 3),
        round(std, 3)),
        fontsize=fontsize
    )

    plt.savefig(os.path.join(dir_path, f'{j}.svg'), transparent=True)
    plt.close()
