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

def _g_recurrence1(i, x):
    return i+1, x+1
x = 1
i,x1 = control_flow_ops.while_loop(
    cond=lambda i, *_: i < 0,
    body=_g_recurrence1,
    loop_vars=( 0,x))
x2 = x1*2
with tf.Session() as session:
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)
    a = session.run(x2)
    print a 