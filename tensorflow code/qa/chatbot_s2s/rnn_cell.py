#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: rnn_cell.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/28 16:14:48
Brief: 
"""
import sys
import math
import numpy as np

from tensorflow.python.ops import rnn_cell
import matplotlib.pyplot as plt  
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell 

linear = core_rnn_cell._linear

def ln(input, s, b, epsilon = 1e-5, max = 1000):
    """ Layer normalizes a 2D tensor along its second axis, which corresponds to batch """
    m, v = tf.nn.moments(input, [1], keep_dims=True)
    normalised_input = (input - m) / tf.sqrt(v + epsilon)
    return normalised_input * s + b



class LNGRUCell(rnn_cell.RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self, num_units, input_size=None, activation=tf.tanh):
    if input_size is not None:
      print("%s: The input_size parameter is deprecated." % self)
    self._num_units = num_units
    self._activation = activation

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    dim = self._num_units
    with tf.variable_scope(scope or type(self).__name__):  # "GRUCell"
      with tf.variable_scope("Gates"):  # Reset gate and update gate.
        # We start with bias of 1.0 to not reset and not update.
        with tf.variable_scope( "Layer_Parameters"):

          s1 = tf.get_variable("s1", shape=[2*dim], dtype=tf.float32)
          s2 = tf.get_variable("s2", shape=[2*dim], dtype=tf.float32)
          s3 = tf.get_variable("s3", shape=[dim],dtype=tf.float32)
          s4 = tf.get_variable("s4", shape=[dim],dtype=tf.float32)
          b1 = tf.get_variable("b1", shape=[2*dim],dtype=tf.float32)
          b2 = tf.get_variable("b2", shape=[2*dim],dtype=tf.float32)
          b3 = tf.get_variable("b3", shape=[dim], dtype=tf.float32)
          b4 = tf.get_variable("b4", shape=[dim], dtype=tf.float32)

        with tf.variable_scope("input_below_"):
            input_below_ = linear([inputs],2 * self._num_units, False)
        input_below_ = ln(input_below_, s1, b1)
        with tf.variable_scope("state_below_"):
            state_below_ = linear([state],2 * self._num_units, False)
        state_below_ = ln(state_below_, s2, b2)
        out =tf.add(input_below_, state_below_)
        r, u = array_ops.split(out, 2, 1)
        r, u = tf.nn.sigmoid(r), tf.nn.sigmoid(u)

      with tf.variable_scope("Candidate"):
          with tf.variable_scope("input_below_x"):
              input_below_x = linear([inputs],self._num_units, False)
          input_below_x = ln(input_below_x, s3, b3)
          with tf.variable_scope("state_below_x"):
              state_below_x = linear([state],self._num_units, False)
          state_below_x = ln(state_below_x, s4, b4)
          c_pre = tf.add(input_below_x,r * state_below_x)
          c = self._activation(c_pre)
      new_h = u * state + (1 - u) * c
    return new_h, new_h

class LNBasicLSTMCell(rnn_cell.RNNCell):
  """Basic LSTM recurrent network cell.
  The implementation is based on: http://arxiv.org/abs/1409.2329.
  We add forget_bias (default: 1) to the biases of the forget gate in order to
  reduce the scale of forgetting in the beginning of the training.
  It does not allow cell clipping, a projection layer, and does not
  use peep-hole connections: it is the basic baseline.
  For advanced models, please use the full LSTMCell that follows.
  """

  def __init__(self, num_units, forget_bias=1.0, input_size=None,
               state_is_tuple=True, activation=tf.tanh):
    """Initialize the basic LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      forget_bias: float, The bias added to forget gates (see above).
      input_size: Deprecated and unused.
      state_is_tuple: If True, accepted and returned states are 2-tuples of
        the `c_state` and `m_state`.  By default (False), they are concatenated
        along the column axis.  This default behavior will soon be deprecated.
      activation: Activation function of the inner states.
    """
    if not state_is_tuple:
      print("%s: Using a concatenated state is slower and will soon be "
                   "deprecated.  Use state_is_tuple=True.", self)
    if input_size is not None:
        print("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._state_is_tuple = state_is_tuple
    self._activation = activation

  @property
  def state_size(self):
    return (LSTMStateTuple(self._num_units, self._num_units)
            if self._state_is_tuple else 2 * self._num_units)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell (LSTM)."""
    with tf.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
      # Parameters of gates are concatenated into one multiply for efficiency.
      if self._state_is_tuple:
        c, h = state
      else:
        c, h = array_ops.split(state, 2, 1)

      s1 = tf.get_variable("s1", shape=[4 * self._num_units], dtype=tf.float32)
      s2 = tf.get_variable("s2", shape=[4 * self._num_units], dtype=tf.float32)
      s3 = tf.get_variable("s3", shape=[self._num_units], dtype=tf.float32)

      b1 = tf.get_variable("b1", shape=[4 * self._num_units], dtype=tf.float32)
      b2 = tf.get_variable("b2", shape=[4 * self._num_units], dtype=tf.float32)
      b3 = tf.get_variable("b3", shape=[self._num_units], dtype=tf.float32)

      with tf.variable_scope("out_1"):
        input_below_ = linear([inputs],4 * self._num_units, False)
      input_below_ = ln(input_below_, s1, b1)
      with tf.variable_scope("out_2"):
        state_below_ = linear([h],4 * self._num_units, False)
      state_below_ = ln(state_below_, s2, b2)
      lstm_matrix = tf.add(input_below_, state_below_)

      i, j, f, o = array_ops.split(lstm_matrix, 4, 1)

      new_c = (c * tf.nn.sigmoid(f) + tf.nn.sigmoid(i) *
               self._activation(j))

      # Currently normalizing c causes lot of nan's in the model, thus commenting it out for now.
      # new_c_ = ln(new_c, s3, b3)
      new_c_ = new_c
      new_h = self._activation(new_c_) * tf.nn.sigmoid(o)

      if self._state_is_tuple:
        new_state = LSTMStateTuple(new_c, new_h)
      else:
        new_state = tf.concat( [new_c, new_h],1)
      return new_h, new_state
