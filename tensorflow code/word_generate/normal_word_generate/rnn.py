#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: rnn.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/07 14:23:11
Brief: 
"""
import tensorflow as tf
import math

class BasicRNNCell(object):
    def __init__(self, name, unit_number, activation=tf.nn.tanh):
        self._name = name
        self._unit_number = unit_number
        self._activation = activation

    @property
    def name(self):
        return self._name

    @property
    def state_size(self):
        return self._unit_number

    @property
    def output_size(self):
        return self._unit_number

    def zero_state(self,batch_size):
        return tf.zeros([batch_size,self._unit_number], dtype=tf.float32)

    #输入已经是经过embedding后的结果
    def __call__(self, inputs, state, input_size):
        """ 
        Args:
            inputs : [batch_size,poem_length,embedding_size]
            state : [batch_size,self._unit_number]
        """

        with tf.variable_scope(self._name) as scope:
            x = tf.concat([state,inputs],1)
            w = tf.get_variable("rnn_w",shape=[x.get_shape()[1], \
                    self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,1.0/math.sqrt(256+512 )))#x.shape
            mean_w = tf.reduce_mean(w)
            variance_w = tf.reduce_mean(tf.square(w-mean_w))
            tf.summary.scalar('mean_w', mean_w)
            tf.summary.scalar('variance_w', variance_w)

            b = tf.get_variable("rnn_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,1.0/math.sqrt(input_size)))
            output = self._activation(tf.matmul(x,w)+b)
            return output,output


class LSTMRNNCell(object):
    def __init__(self, name, unit_number, activation=tf.nn.tanh):
        self._name = name
        self._unit_number = unit_number
        self._activation = activation

    @property
    def name(self):
        return self._name

    @property
    def state_size(self):
        return self._unit_number

    @property
    def output_size(self):
        return self._unit_number

    def zero_state(self,batch_size):
        return (tf.zeros([batch_size,self._unit_number], dtype=tf.float32),tf.zeros([batch_size,self._unit_number], dtype=tf.float32))

    #输入已经是经过embedding后的结果
    def __call__(self, inputs, state, input_size):
        """ 
        Args:
            inputs : [batch_size,poem_length,embedding_size]
            state : [batch_size,self._unit_number]
        """

        with tf.variable_scope(self._name) as scope:
            (s_prev,h_prev) = state
            x = tf.concat([s_prev,inputs],1)

            #遗忘门
            forget_gate_w = tf.get_variable("fg_w",shape=[x.get_shape()[1], \
                    self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            forget_gate_bias = tf.get_variable("fg_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            forget_gate_o = tf.matmul(x,forget_gate_w)+forget_gate_bias

            input_gate_w = tf.get_variable("ig_w",shape=[x.get_shape()[1], \
                    self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_gate_bias = tf.get_variable("ig_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_gate_o = tf.matmul(x,input_gate_w)+input_gate_bias

            input_w = tf.get_variable("input_w",shape=[x.get_shape()[1],self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_bias = tf.get_variable("input_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_o = tf.matmul(x,input_w)+input_bias

            output_gate_w = tf.get_variable("og_w",shape=[x.get_shape()[1],self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            output_gate_bias = tf.get_variable("og_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            output_gate_o = tf.matmul(x,output_gate_w)+output_gate_bias

            s = s_prev * tf.sigmoid(forget_gate_o) + tf.sigmoid(input_gate_o)*self._activation(input_o)
            h = self._activation(s) * tf.sigmoid(output_gate_o)
            
            return h,(s,h)


class GRURNNCell(object):
    def __init__(self, name, unit_number, activation=tf.nn.tanh):
        self._name = name
        self._unit_number = unit_number
        self._activation = activation

    @property
    def name(self):
        return self._name

    @property
    def state_size(self):
        return self._unit_number

    @property
    def output_size(self):
        return self._unit_number

    def zero_state(self,batch_size):
        return (tf.zeros([batch_size,self._unit_number], dtype=tf.float32),tf.zeros([batch_size,self._unit_number], dtype=tf.float32))

    #输入已经是经过embedding后的结果
    def __call__(self, inputs, state, input_size):
        """ 
        Args:
            inputs : [batch_size,poem_length,embedding_size]
            state : [batch_size,self._unit_number]
        """

        with tf.variable_scope(self._name) as scope:
            (s_prev,h_prev) = state
            x = tf.concat([s_prev,inputs],1)

            #遗忘门
            forget_gate_w = tf.get_variable("fg_w",shape=[x.get_shape()[1], \
                    self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            forget_gate_bias = tf.get_variable("fg_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            forget_gate_o = tf.matmul(x,forget_gate_w)+forget_gate_bias

            input_gate_w = tf.get_variable("ig_w",shape=[x.get_shape()[1], \
                    self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_gate_bias = tf.get_variable("ig_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_gate_o = tf.matmul(x,input_gate_w)+input_gate_bias

            input_w = tf.get_variable("input_w",shape=[x.get_shape()[1],self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_bias = tf.get_variable("input_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            input_o = tf.matmul(x,input_w)+input_bias

            output_gate_w = tf.get_variable("og_w",shape=[x.get_shape()[1],self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            output_gate_bias = tf.get_variable("og_b",shape=[self._unit_number], \
                    initializer=tf.truncated_normal_initializer(0,2.0/math.sqrt(self._unit_number*2 + input_size)))
            output_gate_o = tf.matmul(x,output_gate_w)+output_gate_bias

            s = s_prev * tf.sigmoid(forget_gate_o) + tf.sigmoid(input_gate_o)*self._activation(input_o)
            h = self._activation(s) * tf.sigmoid(output_gate_o)
            
            return h,(s,h)

