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

start_token = 'B'
end_token = 'E'
blank_token = 'L'

class PoemGenerateInput(object):
    def __init__(self,batch_size,data_file):
        self.batch_size = batch_size

        poems = []
        word_counter = {start_token:1,end_token:1,blank_token:1}
        with tf.gfile.GFile(data_file, "r") as f:
            for line in  f.readlines():
                line = line.decode("utf-8").replace("\n", "")
                if len(line) < 5:
                    continue
                poem = [start_token] + [word for word in line] + [end_token]
                for word in poem:
                    if not word_counter.has_key(word):
                        word_counter[word] = 1
                    word_counter[word] += 1
                poems.append(poem)
        count_pairs = sorted(word_counter.items(),key=lambda x: (-x[1],x[0]))
        words,counts = list(zip(*count_pairs))
        self.vocab_size = len(words)
        print("words vocab_size = %d" % self.vocab_size)

        self.word_to_id_dict = dict(zip(words,range(self.vocab_size)))
        self.id_to_word_dict = dict(zip(range(self.vocab_size),words))

        data = np.array([[self.word_to_id_dict[word] for word in poem] for poem in poems])
        self.data = data
        self.batch_index = 0

        self.batch_datas = []
        self.batch_targets = []
        self.chunk = data.shape[0] // self.batch_size
        for i in range(self.chunk):
            batch_data = []
            batch_target = []
            for j in range(self.batch_size):
                batch_data.append(self.data[ i * self.batch_size + j])
            length = max(map(len, batch_data))
            x_data = np.full((batch_size, length), self.word_to_id_dict[blank_token], np.int32)
            for row in range(batch_size):
                x_data[row, :len(batch_data[row])] = batch_data[row]
            y_data = np.copy(x_data)
            # y的话就是x向左边也就是前面移动一个
            y_data[:, :-1] = x_data[:, 1:]
            self.batch_datas.append(x_data)
            self.batch_targets.append(y_data)
        


        
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
    inputs = PoemGenerateInput(10,5,"datas/poems_sorted.txt")
    x,y = inputs.get_next_batch()
    print x,y
