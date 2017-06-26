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
import random

default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

start_token = "；".decode("utf-8")
blank_token = 'B'

class PoemGenerateInput(object):
    def __init__(self, data_file, word2id_file, poem_length=7):
        self.word2id = {}
        self.id2word = {}

        self.poem_length = poem_length


        self.poem_length = poem_length

        maxid = 0
        with open(word2id_file, "r") as f:
            for line in f.readlines():
                if not line:
                    break
                line = line.strip().decode('utf-8')
                if len(line) < 1:
                    continue
                infos = line.split("\t")
                self.word2id[infos[0]] = int(infos[1])
                self.id2word[int(infos[1])] = infos[0]
                #print int(infos[1])
                if int(infos[1]) > maxid:
                    maxid = int(infos[1])
        self.word2id[start_token] = maxid+1
        self.id2word[maxid+1] = start_token
        self.word2id[blank_token] = maxid+2
        self.id2word[maxid+2] = blank_token

        self.vocab_size = len(self.word2id)
        print("words vocab_size = %d" % self.vocab_size)

        self.data_ids = []
        with tf.gfile.GFile(data_file, "r") as f:
            for line in  f.readlines():
                line = line.decode("utf-8").replace("\n", "").split("\t")[3].replace("；","")
                if len(line) < 5:
                    continue
                poem = [word for word in line] #5 * (poem_length+1)
                if len(poem) != 28:
                    poems = ""
                    for word in poem:
                        poems += word
                    #print poems
                    continue
                self.data_ids.append([self.word2id[word] for word in poem])

        print("data_set size : %d" % len(self.data_ids))
        self.length_ids = {}
        self.length_ids[poem_length] = []
        self.length_ids[poem_length*2] = []
        self.length_ids[poem_length*3] = []
        self.length_ids[poem_length*4] = []
        self.length_set = [poem_length,poem_length*2,poem_length*3,poem_length*4]
        self.length_set_id = 0
        self.batch_index = 0
        for poem_ids in self.data_ids:
            self.length_ids[poem_length].append((poem_ids[0:poem_length], [self.word2id[start_token]]+poem_ids[0:poem_length]))
            self.length_ids[poem_length*2].append((poem_ids[0:poem_length],[self.word2id[start_token]]+poem_ids[poem_length:poem_length*2]))
            self.length_ids[poem_length*3].append((poem_ids[0:poem_length*2],[self.word2id[start_token]]+poem_ids[poem_length*2:poem_length*3]))
            self.length_ids[poem_length*4].append((poem_ids[0:poem_length*3],[self.word2id[start_token]]+poem_ids[poem_length*3:poem_length*4]))

    def get_next_batch(self,batch_size):
        randomid = random.randint(0,len(self.length_set)-1)
        length_set_id = self.length_ids[self.length_set[randomid]]
        #length_set_id = self.length_ids[self.length_set[1]]
        encoder_inputs, decoder_inputs, targets, weights = [], [], [], []
        for i in range(batch_size):
            #index = random.randint(0,len
            encoder_input,decoder_input = random.choice(length_set_id)
            if encoder_input != None:
                encoder_inputs.append(encoder_input)
            decoder_inputs.append(decoder_input)
            target = decoder_input[1:]+[self.word2id[start_token]]
            targets.append(target)
            #print decoder_input[1:]+[start_token]

            batch_weight = np.ones(len(decoder_input), dtype=np.float32)
            batch_weight[-1] = 0.0
            weights.append(batch_weight)
        return encoder_inputs, decoder_inputs, targets, weights, randomid
    
    def get_next_batch_example(self,batch_size):
        encoder_inputs, decoder_inputs, targets, weights = [], [], [], []
        randomid = random.randint(0,4)
        for i in range(batch_size):
            encoder_input = np.array([1,2,3,4,5]) + 10*i
            decoder_input = np.array([0,6,7,8,9,10]) + 10*i
            target = np.append(decoder_input[1:],0)
            
            encoder_inputs.append(encoder_input)  
            decoder_inputs.append(decoder_input)       
            targets.append(target)    
        return np.array(encoder_inputs), np.array(decoder_inputs), np.array(targets), weights, randomid

if __name__ == '__main__':
    inputs = PoemGenerateInput("datas/poems7_most3000.txt","datas/word2id.txt")
    encoder_inputs, decoder_inputs, targets, weights ,randomid= inputs.get_next_batch(1)
    for i in range(len(encoder_inputs)):
        x_result = ""
        y_result = ""
        if encoder_inputs[i] != None:
            for j in range(len(encoder_inputs[i])):
                x_result += inputs.id2word[encoder_inputs[i][j]]
        for j in range(len(decoder_inputs[i])):
            y_result += inputs.id2word[decoder_inputs[i][j]]
        y_result+="\n"
        for j in range(len(targets[i])):
            y_result += inputs.id2word[targets[i][j]]

        print x_result
        print y_result