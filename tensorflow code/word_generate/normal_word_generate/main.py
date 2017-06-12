#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c)     17 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: main.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/07 12:21:36
Brief: 
"""
import argparse
import sys
import math
import numpy as np
import time
import matplotlib.pyplot as plt  
import os
import tensorflow as tf
 
import datasets
import datasets_poem
import rnn
import model

import sys
reload(sys)
sys.setdefaultencoding('utf-8')



#定义命令行读入参数的默认值

tf.flags.DEFINE_string("run_type", "train",
                "train,inference,....")
tf.flags.DEFINE_string("train_dir", "datas",
                "train_dir")
tf.flags.DEFINE_string("train_file", "datas/raw.txt",
                "train_file")
tf.flags.DEFINE_string("logdir", "logdir",
                "logdir")
tf.flags.DEFINE_string("head", "None",
                "head")
tf.flags.DEFINE_integer("batch_size", 10,
                "batch size")
tf.flags.DEFINE_integer("time_steps", 2,
                "time steps size")
tf.flags.DEFINE_integer("max_iter", 1000,
                "max_iter")
tf.flags.DEFINE_float("learning_rate", 1.0,
                "learning_rate")


class SimpleConfig(object):
    train_dir = tf.flags.FLAGS.train_dir
    train_file = tf.flags.FLAGS.train_file
    run_type = tf.flags.FLAGS.run_type
    batch_size = tf.flags.FLAGS.batch_size
    time_steps = tf.flags.FLAGS.time_steps
    max_iter = tf.flags.FLAGS.max_iter
    learning_rate = tf.flags.FLAGS.learning_rate
    logdir = tf.flags.FLAGS.logdir
    head = tf.flags.FLAGS.head
    
def run_train(config,datasets_,run_iter,end_points,input_data,output_targets,inited=False):
    
    vocab_size = datasets_.vocab_size
    print("vocab_size : %d" % vocab_size)

    saver = tf.train.Saver(tf.global_variables())
    with tf.Session() as session:
        summary_writer = tf.summary.FileWriter(config.logdir)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        checkpoint = tf.train.latest_checkpoint(config.logdir)
        start_epoch = 0
        if checkpoint:
            saver.restore(session, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        start_time = time.time()
        iters = 0
        for iter_i  in range(start_epoch, start_epoch + run_iter):
            print("iter all : %d",start_epoch + run_iter)
            cost_all = 0
            for chunk_i in range(datasets_.chunk):
                x_data,y_data = datasets_.get_next_batch()
                time_steps = len(x_data[0])
                cost,summary_str,_ = session.run(
                        [   
                            end_points['total_loss'],
                            end_points['summary'],
                            end_points['train_op']
                        ],
                        feed_dict=
                        {
                            input_data: x_data, 
                            output_targets: y_data
                        })
            #return
                iters += time_steps
                cost_all += cost
                if chunk_i % 100 == 0 :
                    print("iter_i : %d, chunk %d,flex : %.4f,speed: %.0f wps" % (iter_i,chunk_i,np.exp(cost/time_steps),iters * config.batch_size / (time.time() - start_time)))
                    summary_writer.add_summary(summary_str, chunk_i)
                    summary_writer.flush()
            #print("iter_r : %d, cost_all : %.2f, avg_flex : %.2f" % (iter_i, cost_all, np.exp(cost_all/datasets_.chunk/config.time_steps)) )
            if iter_i % 2 == 0:
                session.run(end_points['learning_rate_decay_op'])
                print("run learning_rate_decay_op")
        saver.save(session, os.path.join(config.logdir, "check_point"), global_step=iter_i)
        print("saving check point")

def to_word(predict, vocabs):
    '''
    predict = predict[0]
    #print(predict)
    maxi = 0
    maxvalue = 0.0
    for i in range(len(predict)):
        if predict[i] > maxvalue:
            maxvalue = predict[i]
            maxi = i
    return vocabs[maxi]
    '''
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]
    


def run_predict(config,datasets_,end_points,input_data,max_len = 100,head=None):
    batch_size = 1
    word2id = datasets_.word_to_id_dict

    saver = tf.train.Saver(tf.global_variables())
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        generated_words = datasets_poem.start_token
        checkpoint = tf.train.latest_checkpoint(config.logdir)
        saver.restore(sess, checkpoint)

        x = np.zeros((1, 1))
        x[0, 0] = word2id[datasets_poem.start_token]
        [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                         feed_dict={input_data: x})
        word = to_word(predict, datasets_.id_to_word_dict)
        
        head_index = 0
        head_flag = True
        for i in range(1,max_len):
            if head != None and head_index < len(head) and head_flag:
                word = head[head_index]
                head_flag = False
                head_index += 1
            if word == '。':
                head_flag = True

            generated_words += word
            x = np.zeros((1, 1))
            x[0, 0] = word2id[word]
            [predict, last_state] = sess.run([end_points['prediction'], end_points['last_state']],
                                             feed_dict={input_data: x, end_points['initial_state']: last_state})
            word = to_word(predict, datasets_.id_to_word_dict)
        return generated_words

def main(_):
    fopen = open("test.out",'w')
    config = SimpleConfig()
    head = config.head.decode('utf-8')
    
    #datasets_ = datasets.WordGenerateInput(config.batch_size,config.time_steps,config.train_file)
    datasets_ = datasets_poem.PoemGenerateInput(config.batch_size,config.train_file)
    word2id = datasets_.word_to_id_dict
    for word in head:
        print word,word2id[word]
    return
    input_data = tf.placeholder(tf.int32, [config.batch_size, None])
    output_targets = tf.placeholder(tf.int32, [config.batch_size, None])
    end_points = model.rnn_model("lstm",input_data,output_targets,datasets_.vocab_size,batch_size=config.batch_size,reused=False)

    predict_input_data = tf.placeholder(tf.int32, [1, None])

    predict_end_points = model.rnn_model("lstm",predict_input_data,None,datasets_.vocab_size,batch_size=config.batch_size,reused=True)

    print("datasets_.chunk : %d" %  datasets_.chunk)
    if config.run_type == 'train':
        for i in range(config.max_iter):
            run_train(config,datasets_,20,end_points,input_data,output_targets)
            predict_result = run_predict(config,datasets_,predict_end_points,predict_input_data,head=head)
            print("generated words : %s" % predict_result)
            fopen.write("generated words : %s\n" % predict_result)
    else:
        print("generated words : %s" % run_predict(config,datasets_,predict_end_points,predict_input_data,head=head))
    fopen.close()

if __name__ == '__main__':
    tf.app.run()
