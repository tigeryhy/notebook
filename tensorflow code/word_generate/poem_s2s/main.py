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
 
import datasets_poem
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
tf.flags.DEFINE_string("head", None,
                "head")
tf.flags.DEFINE_string("poem_first", None,
                "poem_first")
tf.flags.DEFINE_integer("batch_size", 64,
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
    poem_first = tf.flags.FLAGS.poem_first
    
def run_train(m,datasets_,graph,config,m_predict):
    
    vocab_size = datasets_.vocab_size
    print("vocab_size : %d" % vocab_size)
    word2id = datasets_.word2id
    id2word = datasets_.id2word
    fout = open("generate_poem.txt",'w')
    test_words = "；两个黄鹂鸣翠柳".decode("utf-8")
    test_wordsid = [word2id[word] for word in test_words]
    print test_wordsid

    with tf.Session(graph=graph) as session:
        summary_writer = tf.summary.FileWriter(config.logdir)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        session.run(init_op)
        saver = tf.train.Saver(tf.global_variables())
        checkpoint = tf.train.latest_checkpoint(config.logdir)
        start_epoch = 1
        if checkpoint:
            saver.restore(session, checkpoint)
            print("[INFO] restore from the checkpoint {0}".format(checkpoint))
            start_epoch += int(checkpoint.split('-')[-1])
        start_time = time.time()
        print("max_iter : %d " % config.max_iter)
        avg_cost = 0.0 
        while start_epoch < config.max_iter:
            encoder_inputs, decoder_inputs, targets, weights, i = datasets_.get_next_batch(config.batch_size)
            if i == 0:
                cost,_=  session.run([m.cost_single,m.train_op_single],feed_dict=
                                        {
                                            #m.input_encoder_pl : encoder_inputs,
                                            m.input_decoder_pl : decoder_inputs,
                                            m.targets_decoder_pl : targets
                                        })
            else:
                cost,_=  session.run([m.cost,m.train_op],feed_dict=
                                        {
                                            m.input_encoder_pl : encoder_inputs,
                                            m.input_decoder_pl : decoder_inputs,
                                            m.targets_decoder_pl : targets
                                        })
            avg_cost += cost
            cost_epoch = 100
            if start_epoch % cost_epoch == 0:
                print("epoch times : %d, avg flex : %.4f" % (start_epoch,np.exp(avg_cost/cost_epoch)) )
                avg_cost = 0.0 
            
            if start_epoch % 10000 == 1:
                gen_tokens_sg =  session.run(m_predict.gen_tokens_sg,feed_dict=
                                    {
                                        m_predict.input_decoder_pl : [[word2id[datasets_poem.start_token]]]
                                    })
                gen_words = gen_tokens_sg.reshape([-1]).tolist()[1:]
                result = ""
                for wordid in gen_words:
                    result += id2word[wordid]
                result += "，"
                for i in range(3):
                    tmp_result = result
                    tmpout = ""
                    gen_tokens_g,gen_as =  session.run([m_predict.gen_tokens_g,m_predict.gen_as],feed_dict=
                                    {
                                        m_predict.input_encoder_pl : np.reshape(gen_words,[1,-1]),
                                        m_predict.input_decoder_pl : [[word2id[datasets_poem.start_token]]]
                                    })
                    gen_words_s = gen_tokens_g.reshape([-1]).tolist()[1:]
                    for wordid in gen_words_s:
                        result += id2word[wordid]
                        tmpout += id2word[wordid]
                    gen_words = gen_words + gen_words_s
                    result += ","
                    print("encoder words : %s" % tmp_result )
                    print("decoder words : %s" % tmpout )
                    print("gen_as : %s" % gen_as )

                print("gen words : %s" % result)

                fout.write("gen words : %s\n" % result)

                test_result = test_words+","
                gen_words = test_wordsid
                for i in range(3):
                    gen_tokens_g =  session.run(m_predict.gen_tokens_g,feed_dict=
                                    {
                                        m_predict.input_encoder_pl : np.reshape(gen_words,[1,-1]),
                                        m_predict.input_decoder_pl : [[word2id[datasets_poem.start_token]]]
                                    })
                    gen_words_s = gen_tokens_g.reshape([-1]).tolist()[1:]
                    for wordid in gen_words_s:
                        test_result += id2word[wordid]
                    gen_words = gen_words + gen_words_s
                    test_result+=","
                print("test words : %s" % test_result)
                fout.write("test words : %s\n" % test_result)
                fout.flush(); 

            if start_epoch % 10000 == 0:
                print("saving")
                saver.save(session, os.path.join(config.logdir, "check_point"), global_step=start_epoch)
            start_epoch += 1
    fout.close()

def to_word(predict, vocabs):
    ''' 
    print predict
    predict = predict[0]
    #print(predict)
    maxi = 0
    maxvalue = 0.0
    for i in range(len(predict)):
        if predict[i] > maxvalue:
            maxvalue = predict[i]
            maxi = i
    print maxi,maxvalue,vocabs[maxi]
    return vocabs[maxi]
    '''
    t = np.cumsum(predict)
    s = np.sum(predict)
    sample = int(np.searchsorted(t, np.random.rand(1) * s))
    if sample > len(vocabs):
        sample = len(vocabs) - 1
    return vocabs[sample]
     
def main(_):
    config = SimpleConfig()
    graph = tf.Graph()
    datasets_ = datasets_poem.PoemGenerateInput("datas/poems7_most3000.txt","datas/word2id.txt")
    with graph.as_default():
        m = model.Sequence2SequanceModel("lngru",datasets_.vocab_size,batch_size=config.batch_size)
        m.build_attention_decoder_graph()

        m_predict = model.Sequence2SequanceModel("lngru",datasets_.vocab_size,batch_size=1,reuse=True)
        m_predict.build_attention_decoder_graph()

    run_train(m,datasets_,graph,config,m_predict)

if __name__ == '__main__':
    tf.app.run()
