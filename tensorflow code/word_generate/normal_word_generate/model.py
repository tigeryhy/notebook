#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: model.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/07 16:14:48
Brief: 
"""
import sys
import math
import numpy as np

import matplotlib.pyplot as plt  

import tensorflow as tf
from tensorflow.python.ops import array_ops
import rnn


def rnn_model(model, input_data, output_data, vocab_size, rnn_size=256, num_layers=2, batch_size=64,
              learning_rate=0.01,optimizer="Adam",learning_rate_decay_factor=0.5,reused=True):
    """
    construct rnn seq2seq model.
    :param model: model class
    :param input_data: input data placeholder
    :param output_data: output data placeholder
    :param vocab_size:
    :param rnn_size:
    :param num_layers:
    :param batch_size:
    :param learning_rate:
    :return:
    """
    end_points = {}
    with tf.variable_scope("generate_words",reuse=reused):
        learning_rate_var = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)
        learning_rate_decay_op = learning_rate_var.assign(
                learning_rate_var * learning_rate_decay_factor)

        def new_cell():
            if model == 'rnn':
                return tf.contrib.rnn.BasicRNNCell
            elif model == 'gru':
                return tf.contrib.rnn.GRUCell
            elif model == 'lstm':
                return tf.contrib.rnn.BasicLSTMCell
        cell = tf.contrib.rnn.MultiRNNCell([new_cell()(rnn_size, state_is_tuple=True,reuse=reused) for _ in range(num_layers)], state_is_tuple=True)

        if output_data is not None:
            initial_state = cell.zero_state(batch_size, tf.float32)
        else:
            initial_state = cell.zero_state(1, tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable('embedding',[vocab_size, rnn_size],
                initializer=tf.truncated_normal_initializer(stddev=1.0/(vocab_size+rnn_size)))
            inputs = tf.nn.embedding_lookup(embedding, input_data)

        # [batch_size, ?, rnn_size] = [64, ?, 128]
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = tf.reshape(outputs, [-1, rnn_size])

        weights = tf.get_variable(
                "weights", [rnn_size, vocab_size], dtype=tf.float32, \
            initializer=tf.truncated_normal_initializer(stddev=2.0/(rnn_size+rnn_size)))
        bias = tf.get_variable("bias", [vocab_size], dtype=tf.float32, \
                                    initializer=tf.truncated_normal_initializer(stddev=2.0/(rnn_size+rnn_size)))
        logits = tf.matmul(output, weights) + bias # [batch_size, num_step]
        # [?, vocab_size+1]

    

        if output_data is not None:
            # output_data must be one-hot encode
            labels = tf.reshape(output_data, [-1])
            # should be [?, vocab_size+1]

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            # loss shape should be [?, vocab_size+1]
            total_loss = tf.reduce_sum(loss) / batch_size
            
            if optimizer == "SGD":
                optimizer_op = tf.train.GradientDescentOptimizer(learning_rate)
            elif optimizer == "Adam":
                optimizer_op = tf.train.AdamOptimizer(learning_rate)

            #optimizer_op = tf.train.GradientDescentOptimizer(1)
            train_op = optimizer_op.minimize(total_loss)

            tf.summary.scalar("loss",total_loss)
            summary = tf.summary.merge_all()

            end_points['initial_state'] = initial_state
            end_points['output'] = output
            end_points['train_op'] = train_op
            end_points['total_loss'] = total_loss
            end_points['loss'] = loss
            end_points['last_state'] = last_state
            end_points['summary'] = summary
            end_points['learning_rate_decay_op'] = learning_rate_decay_op
        else:
            prediction = tf.nn.softmax(logits)

            end_points['initial_state'] = initial_state
            end_points['last_state'] = last_state
            end_points['prediction'] = prediction

    return end_points
