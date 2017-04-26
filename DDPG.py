import ACNet
from ACNet import Poliy
from ACNet import QValue
from ACNet import ReplayBuffer
from ACNet import OUProcess

import random
from collections import deque

import keras.backend as K
import tensorflow as tf
from keras.engine.training import *
from keras.layers import Dense, Input, merge
from keras.optimizers import Adam

QLEARNINGRATE=0.05
PLEARNINGRATE=0.05
BATCHSIZE=100
#[[angles],open_or_close]
ACTIONSIZE=8
#[[angles],[x_clip,y_clip,z_clip],open_or_close,[x_item,y_item,z_item]]
STATESIZE=14

def DDPG():
