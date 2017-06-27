#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2015 Baidu.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: http_server.py
Author: wangyan
Date: 2015/07/08 17:54:27
Brief: http server (kworks storages)
"""

import re
import sys
import json
import urllib
import urllib2
import tornado
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import binascii
import model
import datasets_poem
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



datasets_ = datasets_poem.PoemGenerateInput("datas/poems7_most3000.txt","datas/word2id.txt")
word2id = datasets_.word2id
id2word = datasets_.id2word



m_predict = model.Sequence2SequanceModel("lngru",datasets_.vocab_size,batch_size=1)
m_predict.build_attention_decoder_graph()

session =  tf.Session() 
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
session.run(init_op)
saver = tf.train.Saver(tf.global_variables())
checkpoint = tf.train.latest_checkpoint("logdir")
start_epoch = 1
if checkpoint:
    saver.restore(session, checkpoint)
    print("[INFO] restore from the checkpoint {0}".format(checkpoint))
    start_epoch += int(checkpoint.split('-')[-1])


class HomeHandler(tornado.web.RequestHandler):
    def get(self):
        url_prefix = "http://demo-ai.roobo.com/inner/query?q="
        query = self.get_argument("query")
        type_ = self.get_argument("type")
        print "query::",query
        if type_ == "head" or type_ == "poem_first":
            if type_ == "head":
                head = query.decode("utf-8")
                length = len(head)
                head_ids = []
                for i in range(len(head)):
                    if word2id.has_key(head[i]):
                        head_ids.append(word2id[head[i]])
                result = ""
                for i in range(len(head_ids)):
                    if i == 0:
                        gen_tokens_sg =  session.run(m_predict.gen_tokens_sg,feed_dict=
                                    {
                                        m_predict.input_decoder_pl : [[word2id[datasets_poem.start_token],head_ids[i]]]
                                    })
                        gen_words = gen_tokens_sg.reshape([-1]).tolist()[1:]
                        result = ""
                        for wordid in gen_words:
                            result += id2word[wordid]
                        result += "，"
                    else:
                        tmp_result = result
                        tmpout = ""
                        gen_tokens_g,gen_as,v =  session.run([m_predict.gen_tokens_g,m_predict.gen_as,m_predict.v],feed_dict=
                                        {
                                            m_predict.input_encoder_pl : np.reshape(gen_words,[1,-1]),
                                            m_predict.input_decoder_pl : [[word2id[datasets_poem.start_token],head_ids[i]]]
                                        })
                        gen_words_s = gen_tokens_g.reshape([-1]).tolist()[1:]
                        for wordid in gen_words_s:
                            result += id2word[wordid]
                            tmpout += id2word[wordid]
                        gen_words = gen_words + gen_words_s
                        result += ","
                print("gen words : %s" % result)
                self.write(result)
            if type_ == "poem_first":
                poem_first = query.decode("utf-8")
                if len(poem_first) != 7:
                    self.write("first poem's length must eques 7")
                    print("first poem's length must eques 7")
                    return
                ids = []
                for i in range(len(poem_first)):
                    if word2id.has_key(poem_first[i]):
                        ids.append(word2id[poem_first[i]])
                    else:
                        print("word's set has no word %s" % poem_first[i])
                        self.write("word's set has no word %s" % poem_first[i])
                        return 
                result = poem_first+"，"
                gen_words = ids
                for i in range(3):
                    tmp_result = result
                    tmpout = ""
                    start_token = word2id[datasets_poem.start_token]
                    gen_tokens_g,gen_as,v =  session.run([m_predict.gen_tokens_g,m_predict.gen_as,m_predict.v],feed_dict=
                                    {
                                        m_predict.input_encoder_pl : np.reshape(gen_words,[1,-1]),
                                        m_predict.input_decoder_pl : [[start_token]]
                                    })
                    gen_words_s = gen_tokens_g.reshape([-1]).tolist()[1:]
                    for wordid in gen_words_s:
                        result += id2word[wordid]
                        tmpout += id2word[wordid]
                    gen_words = gen_words + gen_words_s
                    result += ","
                print("gen words : %s" % result)
                self.write(result)
        else:
            self.write("type not support")


def main(conf_dir, port = 8789):
	#run web server
    app = tornado.web.Application([
        (r"/platform/ai", HomeHandler)
        ],
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(port)
    tornado.ioloop.IOLoop.instance().start()
    return True


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: cmd + port")
        sys.exit(-1)
    port = int(sys.argv[1])
    main("", port)
