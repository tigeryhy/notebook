# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: poems.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/06 18:02:36
Brief: 
"""
import collections
import sys
import time
import numpy as np
import tensorflow as tf

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)


class WordGenerateInput(object):
    def __init__(self,batch_size,num_step,data_file):
        self.batch_size = batch_size
        self.num_step = num_step

        with tf.gfile.GFile(data_file, "r") as f:
            raw_data = f.read().decode("utf-8").replace("\n", " ").split()
        counter = collections.Counter(raw_data)
        count_pairs = sorted(counter.items(),key=lambda x: (-x[1],x[0]))
        words,counts = list(zip(*count_pairs))
        self.vocab_size = len(words)
        print("words vocab_size = %d" % self.vocab_size)

        self.word_to_id_dict = dict(zip(words,range(self.vocab_size)))
        self.id_to_word_dict = dict(zip(range(self.vocab_size),words))
        data = np.array([self.word_to_id_dict[word] for word in raw_data])
        self.batch_length = len(data)//batch_size
        data = data[:batch_size * self.batch_length]
        data = data.reshape(batch_size,self.batch_length)
        self.data = data
        self.batch_index = 0

        self.batch_datas = []
        self.batch_targets = []
        self.chunk = self.batch_length // self.num_step
        #print(self.batch_length,self.num_step,self.chunk)
        for i in range(self.chunk):
            batch_data = []
            batch_target = []
            for j in range(self.batch_size):
                batch_data.append(self.data[j][i * self.num_step : (i+1) * self.num_step])
                batch_target.append(self.data[j][i * self.num_step+1 : (i+1) * self.num_step + 1] )
                # print(i * self.num_step, (i+1) * self.num_step)
                # print(i * self.num_step + 1, (i+1) * self.num_step + 1)
                # print(self.data[j][i * self.num_step : (i+1) * self.num_step])
                # print(self.data[j][i * self.num_step+1 : (i+1) * self.num_step+1])
            self.batch_datas.append(batch_data)
            self.batch_targets.append(batch_target)

        self.batch_datas = np.array(self.batch_datas)
        self.batch_targets = np.array(self.batch_targets)
        
        print(np.shape(self.batch_datas))
        print(np.shape(self.batch_targets))
        # print(self.word_to_id_dict)
        # print(self.batch_datas)
        #print(self.batch_targets)

    def get_next_batch(self):
        if self.batch_index >= self.chunk:
            self.batch_index = 0
        x,y = self.batch_datas[self.batch_index] , self.batch_targets[self.batch_index]
        self.batch_index += 1
        return x,y

    def get_next_random_batch(self):
        #np.random.seed(time.time())
        x = np.array(np.random.randint(low=0,high=self.vocab_size,size=self.batch_size))
        x.reshape(self.batch_size,1)
        return x,x




if __name__ == '__main__':
    inputs = WordGenerateInput(10,5,"datas/raw_test.txt")
    x,y = inputs.get_next_batch()
    print x,y
