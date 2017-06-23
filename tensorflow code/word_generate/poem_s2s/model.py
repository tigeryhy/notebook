#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
# 
# Copyright (c) 2017 roobo.com, Inc. All Rights Reserved
# 
########################################################################
 
"""
File: model.py
Author: yuhaiyang(yuhaiyang@roobo.com)
Date: 2017/06/07 16:14:48
Brief: 
"""
import sys
import math
import numpy as np

import matplotlib.pyplot as plt  
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
import datasets_poem

linear = core_rnn_cell_impl._linear

class Sequence2SequanceModel(object):
    def __init__(self,model,vocab_size, rnn_size=256, num_layers=2, batch_size=64,attention_size=256,
        learning_rate=0.01,optimizer="Adam",tone_size=7,max_grad_norm=5,reuse=False,start_token="；".decode("utf-8")):
        """
        construct rnn seq2seq model.
        :param model: model class
        :param vocab_size:
        :param rnn_size:
        :param num_layers:
        :param batch_size:
        :param learning_rate:
        :return:
        """
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.attention_size = attention_size
        self.batch_size = batch_size
        self.tone_size = tone_size
        self.learning_rate = 0.01
        self.model = model
        self.num_layers = num_layers
        self.max_grad_norm = max_grad_norm
        self.reuse = reuse
        self.start_token = start_token

    

    def build_attention_decoder_graph(self):
        #encoder 和 decoder 公用一个前向rnn
        rnn_size = self.rnn_size
        vocab_size = self.vocab_size 
        
        with tf.variable_scope("generate_poem",reuse=self.reuse,initializer=tf.random_uniform_initializer(-0.08,0.08)):
            def new_cell():
                if self.model == 'rnn':
                    return tf.contrib.rnn.BasicRNNCell
                elif self.model == 'gru':
                    return tf.contrib.rnn.GRUCell
                elif self.model == 'lstm':
                    return tf.contrib.rnn.BasicLSTMCell
            self.cell_forward = tf.contrib.rnn.MultiRNNCell([new_cell()(rnn_size) for i in range(self.num_layers)], state_is_tuple=True)

            self.cell_back = tf.contrib.rnn.MultiRNNCell([new_cell()(rnn_size) for i in range(self.num_layers)], state_is_tuple=True)

            self.cell_decoder =  tf.contrib.rnn.MultiRNNCell([new_cell()(rnn_size) for i in range(self.num_layers)], state_is_tuple=True)

            with tf.device("/cpu:0"):
                self.embedding = tf.get_variable('embedding',[vocab_size, rnn_size],
                    initializer=tf.truncated_normal_initializer(stddev=1.0/(vocab_size+rnn_size)))
                    
            self.input_encoder_pl = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,None])
            self.input_decoder_pl = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,None])
            self.targets_decoder_pl = tf.placeholder(dtype=tf.int32, shape=[self.batch_size,None])
            # self.targets_weight = tf.placeholder(dtype=tf.float32, shape=[self.batch_size,self.tone_size+1])
            input_decoder_pl_trans = tf.transpose(self.input_decoder_pl,[1,0])
            targets_decoder_pl_trans = tf.transpose(self.targets_decoder_pl,[1,0])
            batch_size = self.batch_size
            #encoder
            initial_state_forward = self.cell_forward.zero_state(batch_size, tf.float32)
            initial_state_back = self.cell_back.zero_state(batch_size, tf.float32)
            
            encoder_embedding_inputs = tf.transpose(tf.nn.embedding_lookup(self.embedding, self.input_encoder_pl),[1,0,2]) #[time_steps,batch_size,embedding_size]
            time_steps = tf.shape(encoder_embedding_inputs)[0]

            encoder_inputs = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=time_steps,
                                        tensor_array_name="encoder_inputs")
            encoder_inputs1 = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=time_steps,
                                        tensor_array_name="encoder_inputs1")
            encoder_inputs = encoder_inputs.unstack(encoder_embedding_inputs)
            encoder_inputs1 = encoder_inputs1.unstack(encoder_embedding_inputs)

            encoder_outputs_forward = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=time_steps,
                                        tensor_array_name="encoder_outputs_forward")
            encoder_outputs_back = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=time_steps,
                                        tensor_array_name="encoder_outputs_back")


            def _birnn_recurrence(i, s_forward_pre,s_back_pre,
                                  encoder_outputs_forward_ta,encoder_outputs_back_ta):
                xt_pre = encoder_inputs.read(i)
                xt_back = encoder_inputs1.read(time_steps-1-i)
                with tf.variable_scope("rnn_forward"):
                    h_forward,s_forward = self.cell_forward(xt_pre,s_forward_pre)
                encoder_outputs_forward_ta = encoder_outputs_forward_ta.write(i,h_forward)

                h_back,s_back = self.cell_back(xt_back,s_back_pre,scope="rnn_back")
                encoder_outputs_back_ta = encoder_outputs_back_ta.write(time_steps-1-i,h_back)
                #encoder_outputs_back_ta.write(i,h_back)
                return i + 1, s_forward, s_back, \
                       encoder_outputs_forward_ta, encoder_outputs_back_ta

            tmpi,forward_statu,back_statu,encoder_outputs_forward,encoder_outputs_back= control_flow_ops.while_loop(
                cond = lambda i,*_ : i < time_steps,
                body = _birnn_recurrence,
                loop_vars = (0, initial_state_forward, initial_state_back
                    ,encoder_outputs_forward,encoder_outputs_back)
            )
            
            encoder_outputs_forward = encoder_outputs_forward.stack()
            encoder_outputs_back = encoder_outputs_back.stack()
            encoder_outputs = tf.concat([encoder_outputs_forward,encoder_outputs_back],2)

            #[h_pre,
            # h_back]
            
            encoder_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
            #encoder_outputs
            #[batch_size, time_step, rnn_size*2]
            
            '''
            decoder:
                h(t) = f(h(t-1),s(t-1),c(t))
            attention:
                c = sum(softmax(g(s(t-1),encoder_outputs)) * encoder_outputs) 
                g(s(t-1),encoder_outputs) = v * tanh(W*s(t-1) + U*encoder_outputs)) : [batch_size,time_steps]
            '''
            
            decoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_decoder_pl)
            decoder_time_steps = tf.shape(decoder_embedding_inputs)[1]

            encoder_hidden_size = 2 * self.rnn_size
            attention_length = tf.shape(encoder_outputs)[1]
            attention_hidden_size = encoder_hidden_size #这里将attention的hidden_size设为和encoder_output的hidden_size一样大

            #U*encoder_outputs 可以预先计算, 这里用卷积的方式来计算 [batch_size,attention_length,encoder_hidden_size]-> [batch_size,attention_length,attention_hidden_size]
            encoder_outputs_attention = tf.reshape(encoder_outputs,[-1,attention_length,1,encoder_hidden_size]) #[batch_size,attention_length,1,encoder_hidden_size]
            k = tf.get_variable("AttnW",[1, 1, encoder_hidden_size, attention_hidden_size])
            u_mult_encoderout = tf.nn.conv2d(encoder_outputs_attention,k,[1,1,1,1],"SAME")#[batch_size,attention_length,1,attention_hidden_size]
            v = tf.get_variable("AttnV", [attention_hidden_size])
            self.v = v
            def attention(state,reuse=False):
                with tf.variable_scope("attention",reuse=reuse):
                    #W*s(t-1)
                    w_mult_s = linear(state,attention_hidden_size,True) #[batch_size,attention_hidden_size]
                    w_mult_s = tf.reshape(w_mult_s,[-1,1,1,attention_hidden_size])#[batch_size,1,1,attention_hidden_size]
                    g = v * tf.tanh(w_mult_s + u_mult_encoderout) #[batch_size,attention_length,1,attention_hidden_size]
                    g = tf.reduce_sum(g,[2, 3]) ##[batch_size,attention_length]

                    a = tf.nn.softmax(g)#[batch_size,attention_length]
                    a = tf.reshape(a,[-1,attention_length,1,1]) #[batch_size,attention_length,1,1]
                    ah = a * encoder_outputs_attention #[batch_size,attention_length,1,attention_hidden_size]
                    c = tf.reduce_sum(ah,[1,2]) #[batch_size,attention_hidden_size]
                    return c,a

            gen_output = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=decoder_time_steps-1,
                                        tensor_array_name="gen_output")

            decoder_embedding_tensor_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=decoder_time_steps,
                                        tensor_array_name="decoder_embedding_tensor_array")
            decoder_embedding_tensor_array = decoder_embedding_tensor_array.unstack(tf.transpose(decoder_embedding_inputs,[1,0,2]))

            decoder_tensor_array = tensor_array_ops.TensorArray(dtype=tf.int32,
                                        size=decoder_time_steps,
                                        tensor_array_name="decoder_tensor_array")
            decoder_tensor_array = decoder_tensor_array.unstack(tf.transpose(self.input_decoder_pl,[1,0]))
            
            encoder_inputs = encoder_inputs.unstack(encoder_embedding_inputs)
            atten,_ = attention(forward_statu)

            

            #train


            def _decoder_rnn_recurrence(i, s_pre, gen_output,c):
                
                xt = decoder_embedding_tensor_array.read(i)
                with tf.variable_scope("decoder_attention"):
                    xt_c = linear([xt,c],self.rnn_size,True)
                with tf.variable_scope("rnn_decoder"):
                    h_now,s_now = self.cell_decoder(xt_c,s_pre) # h_now = [batch_size, rnn_size]
                    #!!!!!train和gen的答案不一样！！！
                c,a = attention(s_now,True)

                with tf.variable_scope("AttnOutputProjection"):
                    output = linear([h_now,c], self.rnn_size, True)

                with tf.variable_scope("decoder_projection"):
                    w = tf.get_variable(name="out_projection",shape=[self.rnn_size,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    b = tf.get_variable(name="out_bias",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    y_logits = tf.matmul(output,w)+b #这里的h_now可以加上attention的vec?
                    softmax = tf.nn.softmax(y_logits)
                gen_output = gen_output.write(i,softmax)
                
                return i + 1, s_now, gen_output, c

            _,last_statu,gen_output,_= control_flow_ops.while_loop(
                cond = lambda i,*_ : i < decoder_time_steps - 1, 
                body = _decoder_rnn_recurrence,
                loop_vars = (0,forward_statu,gen_output,atten)
            )
            
            #softmax= tf.transpose(gen_output.stack(),[1,0,2]) #[batch_size, decoder_time_steps, vol_size]
            softmax = gen_output.stack()
            
            onehot = tf.one_hot(tf.reshape(targets_decoder_pl_trans[:-1], [-1]), self.vocab_size, 1.0, 0.0)
            log_value = tf.log(
                        tf.clip_by_value(tf.reshape(softmax, [-1, self.vocab_size]), 1e-20, 1.0)
                        )
            self.cost = -tf.reduce_sum(onehot * log_value) / ((self.tone_size) * self.batch_size)
            self.optimizer = tf.train.AdamOptimizer(0.01)
            tvars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), self.max_grad_norm)
            self.train_op = self.optimizer.apply_gradients(zip(grads, tvars))

            #generator

            

            gen_tokens_g = tensor_array_ops.TensorArray(dtype=tf.int32,
                                size=self.tone_size+1,
                                tensor_array_name="gen_tokens")
            def _g_recurrence1(i, s_pre, gen_tokens_g, c):
                
                token = decoder_tensor_array.read(i)
                #xt = tf.nn.embedding_lookup(self.embedding, token)
                xt = decoder_embedding_tensor_array.read(i)
                with tf.variable_scope("decoder_attention",reuse=True):
                    xt_c = linear([xt,c],self.rnn_size,True)
                with tf.variable_scope("rnn_decoder",reuse=True):
                    h_now,s_now = self.cell_decoder(xt_c,s_pre) # h_now = [batch_size, rnn_size]
                c,a = attention(s_now,True)

                gen_tokens_g = gen_tokens_g.write(i,token)
                return i+1, s_now, gen_tokens_g, c
            
            index_i_g, current_state, gen_tokens_g, atten= control_flow_ops.while_loop(
                cond=lambda i, *_: i < decoder_time_steps-1,
                body=_g_recurrence1,
                loop_vars=( 0,forward_statu,gen_tokens_g,atten))

            gen_tokens_g = gen_tokens_g.write(index_i_g,decoder_tensor_array.read(index_i_g))

            def _g_recurrence2(i, xt, s_pre, gen_tokens_g, c):
                
                with tf.variable_scope("decoder_attention",reuse=True):
                    xt_c = linear([xt,c],self.rnn_size,True)
                with tf.variable_scope("rnn_decoder",reuse=True):
                    h_now,s_now = self.cell_decoder(xt_c,s_pre) # h_now = [batch_size, rnn_size]
                c,a = attention(s_now,True)

                with tf.variable_scope("AttnOutputProjection",reuse=True):
                    output = linear([h_now,c], self.rnn_size, True)

                with tf.variable_scope("decoder_projection",reuse=True):
                    w = tf.get_variable(name="out_projection",shape=[self.rnn_size,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    b = tf.get_variable(name="out_bias",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    y_logits = tf.matmul(output,w)+b #这里的h_now可以加上attention的vec?
                    softmax = tf.nn.softmax(y_logits)
                    log_prob = tf.log(softmax)
                    next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                    gen_tokens_g = gen_tokens_g.write(i,next_token)
                x_tp1 = tf.nn.embedding_lookup(self.embedding, next_token)  # batch x emb_dim

                return i+1, x_tp1, s_now, gen_tokens_g, c 

            _, _, _, gen_tokens_g,_ = control_flow_ops.while_loop(
                cond=lambda i, *_: i < self.tone_size+1,
                body=_g_recurrence2,
                loop_vars=( index_i_g+1,tf.nn.embedding_lookup(self.embedding, input_decoder_pl_trans[-1]), 
                            current_state, gen_tokens_g, atten))

            self.gen_tokens_g = gen_tokens_g.stack() #[self.tone_size+1,batch_size]


            #simple decoder            
            initial_state = self.cell_decoder.zero_state(batch_size, tf.float32)

            gen_output_simple = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=decoder_time_steps-1,
                                        tensor_array_name="gen_output_simple")
            def _singel_decoder_rnn_recurrence(i, s_pre, gen_output_simple):
                xt = decoder_embedding_tensor_array.read(i)
                with tf.variable_scope("rnn_decoder",reuse=True):
                    h_now,s_now = self.cell_decoder(xt,s_pre) # h_now = [batch_size, rnn_size]
                
                with tf.variable_scope("decoder_projection",reuse=True):
                    w = tf.get_variable(name="out_projection",shape=[self.rnn_size,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    b = tf.get_variable(name="out_bias",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    y_logits = tf.matmul(h_now,w)+b #这里的h_now可以加上attention的vec?
                    softmax = tf.nn.softmax(y_logits)
                gen_output_simple = gen_output_simple.write(i,softmax)
                return i + 1, s_now, gen_output_simple
            
                
            
            _,last_statu_single,gen_output_simple = control_flow_ops.while_loop(
                cond = lambda i,*_ : i < decoder_time_steps-1,
                body = _singel_decoder_rnn_recurrence,
                loop_vars = (0,initial_state, gen_output_simple)
            )
            #softmax_single= tf.transpose(gen_output_simple.stack(),[1,0,2]) #[batch_size, decoder_time_steps, vol_size]
            softmax_single = gen_output_simple.stack()
            onehot_single = tf.one_hot(tf.reshape(targets_decoder_pl_trans[:-1], [-1]), self.vocab_size, 1.0, 0.0)
            log_value_single = tf.log(
                        tf.clip_by_value(tf.reshape(softmax_single, [-1, self.vocab_size]), 1e-20, 1.0)
                        )
            self.cost_single = -tf.reduce_sum(onehot_single * log_value_single) / ((self.tone_size) * self.batch_size)
            tvars_single = tf.trainable_variables()
            grads_single, _ = tf.clip_by_global_norm(tf.gradients(self.cost_single, tvars_single), self.max_grad_norm)
            self.train_op_single = self.optimizer.apply_gradients(zip(grads_single, tvars_single))


            #simple generator
            #generate first sentence
            initial_state_sg = self.cell_decoder.zero_state(batch_size, tf.float32)
            gen_tokens_sg = tensor_array_ops.TensorArray(dtype=tf.int32,
                                size=self.tone_size+1,
                                tensor_array_name="gen_tokens")
            def _sg_recurrence1(i, s_pre, gen_tokens_sg):
                xt = decoder_embedding_tensor_array.read(i)
                gen_tokens_sg = gen_tokens_sg.write(i,decoder_tensor_array.read(i))
                with tf.variable_scope("rnn_decoder",reuse=True):
                    h_now,s_now = self.cell_decoder(xt,s_pre) # h_now = [batch_size, rnn_size]
                return i+1, s_now, gen_tokens_sg

            index_i, current_state,gen_tokens_sg = control_flow_ops.while_loop(
                cond=lambda i, *_: i < decoder_time_steps-1,
                body=_sg_recurrence1,
                loop_vars=( 0,initial_state_sg,gen_tokens_sg))
            gen_tokens_sg = gen_tokens_sg.write(index_i,decoder_tensor_array.read(index_i))
            def _sg_recurrence2(i, xt, s_pre, gen_tokens):
                with tf.variable_scope("rnn_decoder",reuse=True):
                    h_now,s_now = self.cell_decoder(xt,s_pre) # h_now = [batch_size, rnn_size]
                with tf.variable_scope("decoder_projection",reuse=True):
                    w = tf.get_variable(name="out_projection",shape=[self.rnn_size,self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    b = tf.get_variable(name="out_bias",shape=[self.vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(self.rnn_size+self.vocab_size)))
                    y_logits = tf.matmul(h_now,w)+b #这里的h_now可以加上attention的vec?
                    softmax = tf.nn.softmax(y_logits)
                    log_prob = tf.log(softmax)
                    next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [self.batch_size]), tf.int32)
                    gen_tokens = gen_tokens.write(i,next_token)
                x_tp1 = tf.nn.embedding_lookup(self.embedding, next_token)  # batch x emb_dim
                return i+1, x_tp1, s_now, gen_tokens

            _, _, _, gen_tokens_sg = control_flow_ops.while_loop(
            cond=lambda i, *_: i < self.tone_size+1,
            body=_sg_recurrence2,
            loop_vars=( index_i+1,tf.nn.embedding_lookup(self.embedding, input_decoder_pl_trans[-1]), 
                        current_state, gen_tokens_sg))

            self.gen_tokens_sg = gen_tokens_sg.stack() #[self.tone_size+1,batch_size]



if __name__ == '__main__':
    batch_size = 1
    graph = tf.Graph()
    with graph.as_default():
        m = Sequence2SequanceModel("gru",11,batch_size=batch_size,rnn_size=5,attention_size=5,tone_size=5)
        m.build_attention_decoder_graph()
    datasets_ = datasets_poem.PoemGenerateInput("datas/poems7_most3000.txt","datas/word2id.txt")
    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        type_ = "predict"
        number = 0
        avg_cost = 0.0
        while True:
            number += 1
            encoder_inputs, decoder_inputs, targets, weights, i = datasets_.get_next_batch_example(batch_size)
            #i = 0 
            #i == 0的时候，生成第一句话不需要用到encoder和attention
            #i = 1
            if i == 0:
                cost,_=  session.run([m.cost_single,m.train_op_single],feed_dict=
                                        {
                                            m.input_decoder_pl : decoder_inputs,
                                            m.targets_decoder_pl : targets
                                        })
                #print(" i == 0 , flex : %.4f" % np.exp(cost))
            else:
                cost,_=  session.run([m.cost,m.train_op],feed_dict=
                                        {
                                            m.input_encoder_pl : encoder_inputs,
                                            m.input_decoder_pl : decoder_inputs,
                                            m.targets_decoder_pl : targets
                                        })
                #print(" i != 0 , flex : %.4f" % np.exp(cost))
            avg_cost += cost

            if number % 100 == 0:
                print("flex : %.4f" % np.exp(avg_cost/100))
                avg_cost = 0.0
                gen_tokens_g =  session.run(m.gen_tokens_g,feed_dict=
                                        {
                                            m.input_encoder_pl : [[1,2,3,4,5]],
                                            m.input_decoder_pl : [[0,6]]
                                            #m.input_decoder_pl : [[0,6,7,8,9]]
                                        })
                gen_words = gen_tokens_g.reshape([-1])
                print gen_words

                gen_tokens_sg =  session.run(m.gen_tokens_sg,feed_dict=
                                        {
                                            #m.input_encoder_pl : [[1,2,3,4,5]],
                                            m.input_decoder_pl : [[0,6]]
                                            #m.input_decoder_pl : [[0,6,7,8,9]]
                                        })
                gen_words = gen_tokens_sg.reshape([-1])
                print gen_words

                