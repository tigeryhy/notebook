#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: simple_predict_next_model.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/08 11:39:12
Brief: 
"""
import sys
import math
import numpy as np

import matplotlib.pyplot as plt  

import tensorflow as tf
from tensorflow.python.ops import array_ops
import rnn

class SimplePredictModel(object):
    def __init__(   self,
                    datasets,
                    embbeding_size=200, 
                    batch_size=50, 
                    learning_rate=0.1,
                    learning_rate_decay_factor=0.5):
        self._datasets = datasets
        self._vocab_size = datasets.vocab_size()
        self._embbeding_size = embbeding_size
        self._batch_size = batch_size

        self._learning_rate_decay_factor = learning_rate_decay_factor
        self._learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32)

    def learning_rate_decay_op(self):
        return self._learning_rate.assign(self._learning_rate * self._learning_rate_decay_factor)

    def build_graph(self):
        batch_inputs_pl = tf.placeholder(dtype=tf.int32,shape=[self._batch_size,None],name="origin_input")
        batch_targets_pl = tf.placeholder(dtype=tf.int32,shape=[self._batch_size,None],name="origin_targets")
        embedding = tf.get_variable(
            "embedding", [self._vocab_size, self._embbeding_size],
            initializer=tf.truncated_normal_initializer(stddev=1.0/self._vocab_size),
            dtype=tf.float32)
        batch_inputs_embedded = tf.nn.embedding_lookup(embedding, batch_inputs_pl)
        batch_targets_embedded = tf.nn.embedding_lookup(embedding, batch_targets_pl)

        #让cell的输出和embedding的维数相同
        cell = rnn.BasicRNNCell("basic_cell",self._embbeding_size,activation=tf.nn.relu)
        state = cell.zero_state(self._batch_size)

        outputs = []
        with tf.variable_scope("RNN"):
            #这里不能用这样的方法进行循环，因为batch_inputs_pl是placeholder
            #需要调用tensorflow的接口control_flow_ops.while_loop
            for time_step in range(array_ops.shape(batch_inputs_pl)[1]):
                if time_step > 0: 
                    tf.get_variable_scope().reuse_variables()
                output,state = cell(batch_inputs_embedded[:,time_step,:], state)
                outputs.append(output)
        #outputs : [length,batch_size,_embbeding_size] => [batch_size*length,_embbeding_size]
        outputs = tf.reshape(tf.stack(values = outputs,axis=1),[-1,self._embbeding_size])
        softmax_w = tf.get_variable(
            "softmax_w", [self._embbeding_size, vocab_size], dtype=tf.float32, \
            initializer=tf.truncated_normal_initializer(stddev=1.0/self._embbeding_size))
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32, \
                                    initializer=tf.truncated_normal_initializer(stddev=1.0/self._embbeding_size))
        logits = tf.matmul(outputs, softmax_w) + softmax_b #logits:[batch_size*length,vocab_size]
        targets = tf.reshape(batch_targets_pl, [-1])
        loss = tf.nn_ops.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        self._cost = tf.reduce_sum(loss) / self._batch_size
        self._train_op = tf.train.GradientDescentOptimizer(self._learning_rate).minimize(self._cost)

        tf.summary.scalar("cost",self._cost)
        self._summary = tf.summary.merge_all()

    def train(self, max_epoch=1000, logdir="logdir_simple_predict_next_model"):
        saver = tf.train.Saver(tf.global_variables())
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
            # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
            sess.run(init_op)
            start_epoch = 0
            checkpoint = tf.train.latest_checkpoint(logdir)
            if checkpoint:
                saver.restore(sess, checkpoint)
                print("[INFO] restore from the checkpoint {0}".format(checkpoint))
                start_epoch += int(checkpoint.split('-')[-1])
            print('[INFO] start training...')
            try:
                for epoch in range(start_epoch, max_epoch):
                    n = 0
                    n_chunk = len(poems_vector) // FLAGS.batch_size
                    for batch in range(n_chunk):
                        cost,_,summary = sess.run([self._cost,
                        self._train_op,
                        self._summary
                        ], feed_dict={input_data: batches_inputs[n], output_targets: batches_outputs[n]})
                        n += 1
                        print('[INFO] Epoch: %d , batch: %d , training loss: %.6f' % (epoch, batch, loss))
                    if epoch % 6 == 0:
                        saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
            except KeyboardInterrupt:
                print('[INFO] Interrupt manually, try saving checkpoint for now...')
                saver.save(sess, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix), global_step=epoch)
                print('[INFO] Last epoch were saved, next time will start from epoch {}.'.format(epoch))

    @property
    def cost(self):
        return self._cost

    @property
    def summary(self):
        return self._summary