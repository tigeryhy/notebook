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

len_dict = {}
with open("poems.txt", "r") as f:
    number = 1
    for line in f.readlines():
        if not line:
            break
        line = line.strip()
        if len(line) < 5:
            continue
        if not "。" in line:
            continue
        len_dict[line] = len(line)

    count_pairs = sorted(len_dict.items(),key=lambda x: (x[1],x[0]))
    for count_pair in count_pairs:
        print count_pair[0]

