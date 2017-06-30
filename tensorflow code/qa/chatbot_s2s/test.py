#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: test.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/15 16:07:38
Brief: 
"""

import sys
import math
import numpy as np

import matplotlib.pyplot as plt  

import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops


filename_queue = tf.train.string_input_producer(["data/test1.txt", "data/test2.txt"])
reader = tf.TextLineReader()
key = reader.read(filename_queue)

print tf.Session().run(key)