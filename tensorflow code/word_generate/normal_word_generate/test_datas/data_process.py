#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: data_process.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/06 18:15:07
Brief: 
"""
import collections
import os
import sys

fout_train = open("poems_train.txt","w")
fout_test = open("poems_test.txt","w")
with open("poems.txt", "r") as f:
    number = 1
    for line in f.readlines():
        if not line:
            break
        line = line.strip()
        # words = line.decode("utf-8")
        # for word in words:
        #     print word
        if len(line) < 5 : 
            continue
        if number % 10 == 0:
            fout_test.write(line+"\n")
        else:
            fout_train.write(line+"\n")
        number += 1

fout_train.close()
fout_test.close()


