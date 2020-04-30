import torch
import numpy as np

def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = torch.DoubleTensor(shape).uniform_(-scale, scale)
   # initial = tf.random_uniform(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return initial


def weight_variable_glorot(input_dim, output_dim, name=""):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    
    initial = torch.DoubleTensor(input_dim, output_dim).uniform_(-init_range, init_range)
 #   initial = tf.random_uniform([input_dim, output_dim], minval=-init_range,
  #                              maxval=init_range, dtype=tf.float32)
    return initial

def zeros(shape, name=None):
    """All zeros."""
    initial = torch.zeros(shape, dtype=torch.float32)
    #initial = tf.zeros(shape, dtype=tf.float32) 
    return initial


def ones(shape, name=None):
    """All ones."""
    initial = torch.ones(shape, dtype=torch.float32)
    return initial

