# -*- coding: utf-8 -*-
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils

import rnn_cell
from rnn_cell import LNGRUCell,LNBasicLSTMCell
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.contrib.rnn.python.ops import core_rnn_cell 
from config import FLAGS,BUCKETS


linear = core_rnn_cell._linear
class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, vocab_size, buckets, size,
               num_layers, max_gradient_norm, batch_size,  cell_type="lngru",
               num_samples=512, forward_only=False):
    """Create the model.

    Args:
      vocab_size: size of the source vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
    """
    self.bucket_index_d = {}
    self.vocab_size = vocab_size
    self.buckets = buckets
    

    # Create the internal multi-layer cell for our RNN.
    def new_cell():
      if cell_type == 'rnn':
          return tf.contrib.rnn.BasicRNNCell
      elif cell_type == 'gru':
          return tf.contrib.rnn.GRUCell
      elif cell_type == 'lstm':
          return tf.contrib.rnn.BasicLSTMCell
      elif cell_type == "lngru":
          return LNGRUCell
      elif cell_type == "lnlstm":
          return LNBasicLSTMCell
    cell_encoder = tf.contrib.rnn.MultiRNNCell([new_cell()(size) for i in range(num_layers)])
    cell_decoder =  tf.contrib.rnn.MultiRNNCell([new_cell()(size) for i in range(num_layers)])

    self.input_encoder_pl = tf.placeholder(dtype=tf.int32, shape=[None,None]) #[T,batch_size]
    self.input_decoder_pl = tf.placeholder(dtype=tf.int32, shape=[None,None]) #[T,batch_size]
    self.targets_decoder_pl = tf.placeholder(dtype=tf.int32, shape=[None,None]) #[T,batch_size]
    self.targets_weight_pl = tf.placeholder(dtype=tf.float32, shape=[None,None]) #[T,batch_size]
    
    encoder_time_steps, batch_size = tf.shape(self.input_encoder_pl)[0], tf.shape(self.input_encoder_pl)[1]
    
    #encoder
    with tf.device("/cpu:0"):
      self.embedding = tf.get_variable('embedding',[vocab_size, size],
          initializer=tf.random_uniform_initializer(-0.08,0.08))
    encoder_embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_encoder_pl) #[encoder_time_steps,batch_size,embedding_size]
    encoder_inputs_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                          size=encoder_time_steps,
                          tensor_array_name="encoder_inputs")
    encoder_inputs_array = encoder_inputs_array.unstack(encoder_embedding_inputs)
    initial_state = cell_encoder.zero_state(batch_size, tf.float32)
    encoder_outputs_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                        size=encoder_time_steps,
                                        tensor_array_name="encoder_outputs_forward")


    def _rnn_recurrence(i, s_tm1,encoder_outputs_array):
      xt = encoder_inputs_array.read(i)

      with tf.variable_scope("rnn_forward",reuse=False):
          h_t,s_t = cell_encoder(xt,s_tm1)
      encoder_outputs_array = encoder_outputs_array.write(i,h_t)
      return i + 1, s_t, encoder_outputs_array

    tmpi, encoder_final_state, encoder_outputs_array= control_flow_ops.while_loop(
        cond = lambda i,*_ : i < encoder_time_steps,
        body = _rnn_recurrence,
        loop_vars = (0, initial_state, encoder_outputs_array),
        swap_memory = True
      )


    encoder_outputs = encoder_outputs_array.stack()
    encoder_outputs = tf.transpose(encoder_outputs, perm=[1, 0, 2])
    
    
    #decoder
    decoder_time_steps =  tf.shape(self.input_decoder_pl)[0]
    attention_length = tf.shape(encoder_outputs)[1]
    attention_hidden_size = size
    encoder_outputs_attention = tf.reshape(encoder_outputs,[-1,attention_length,1,size]) #[batch_size,attention_length,1,encoder_hidden_size]
    k = tf.get_variable("AttnW",[1, 1, size, attention_hidden_size])
    u_mult_encoderout = tf.nn.conv2d(encoder_outputs_attention,k,[1,1,1,1],"SAME")#[batch_size,attention_length,1,attention_hidden_size]
    v = tf.get_variable("AttnV", [attention_hidden_size])

    decoder_embedding_array = tensor_array_ops.TensorArray(dtype=tf.float32,
                                size=decoder_time_steps,
                                tensor_array_name="decoder_embedding_array")
    gen_output = tensor_array_ops.TensorArray(dtype=tf.float32,
                                size=decoder_time_steps,
                                tensor_array_name="gen_output")
    decoder_embedding_array = decoder_embedding_array.unstack(tf.nn.embedding_lookup(self.embedding, self.input_decoder_pl))


    def attention(state,reuse=False):
      with tf.variable_scope("attention",reuse=reuse):
        #W*s(t-1)
        w_mult_s = linear(state,attention_hidden_size,False) #[batch_size,attention_hidden_size]
        w_mult_s = tf.reshape(w_mult_s,[-1,1,1,attention_hidden_size])#[batch_size,1,1,attention_hidden_size]
        g = v * tf.tanh(w_mult_s + u_mult_encoderout) #[batch_size,attention_length,1,attention_hidden_size]
        g = tf.reduce_sum(g,[2, 3]) ##[batch_size,attention_length]

        a1 = tf.nn.softmax(g)#[batch_size,attention_length]
        a = tf.reshape(a1,[-1,attention_length,1,1]) #[batch_size,attention_length,1,1]
        ah = a * encoder_outputs_attention #[batch_size,attention_length,1,attention_hidden_size]
        c = tf.reduce_sum(ah,[1,2]) #[batch_size,attention_hidden_size]
        return c,a1
      
    atten,_ = attention(list(encoder_final_state))

    w = tf.get_variable(name="out_projection",shape=[size,vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=2.0/(size+vocab_size)))
    w_t = tf.transpose(w)
    b = tf.get_variable(name="out_bias",shape=[vocab_size],dtype=tf.float32,initializer=tf.truncated_normal_initializer(mean=0.0,stddev=1.0))

    #train
    def _decoder_rnn_recurrence(i, s_pre, gen_output,c):
      xt = decoder_embedding_array.read(i)
      with tf.variable_scope("decoder_attention"):
        xt_c = linear([xt,c],size,False)
      with tf.variable_scope("rnn_decoder"):
        h_now,s_now = cell_decoder(xt_c,s_pre) # h_now = [batch_size, rnn_size]

      c,a = attention(list(s_now),True)
      with tf.variable_scope("AttnOutputProjection"):
        output = linear(list(s_now)+[xt,c], size,True)
      
      cross_entropy = tf.nn.sampled_softmax_loss(w_t, b, tf.reshape(self.targets_decoder_pl[i],[-1,1]), output, num_samples,vocab_size)
      cross_entropy = self.targets_weight_pl[i] * cross_entropy
      gen_output = gen_output.write(i,cross_entropy)
      return i + 1, s_now, gen_output, c


    _,last_statu,gen_output,c= control_flow_ops.while_loop(
        cond = lambda i,*_ : i < decoder_time_steps ,
        body = _decoder_rnn_recurrence,
        loop_vars = (0,encoder_final_state,gen_output,atten),
        swap_memory = True
        )
    gen_output = gen_output.stack() #[decoder_time_steps,batch_size]
    self.gen_output = gen_output
    cross_entropy_all = tf.reduce_sum(gen_output)
    cross_entropy_all += 1e-12
    self.loss = cross_entropy_all / tf.reduce_sum(self.targets_weight_pl)

    optimizer = tf.train.AdamOptimizer(0.01)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), max_gradient_norm)
    self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    self.saver = tf.train.Saver(tf.global_variables())

    #chat generator
    gen_tokens_g = tensor_array_ops.TensorArray(dtype=tf.int32,
                        size=self.buckets[-1][1],
                        tensor_array_name="gen_tokens")
    gen_as = tensor_array_ops.TensorArray(dtype=tf.float32,
                        size=self.buckets[-1][1],
                        tensor_array_name="gen_as")
    def _generator_rnn_recurrence(i, xt, s_pre, gen_tokens_g, c, gen_as):
      with tf.variable_scope("decoder_attention",reuse=True):
          xt_c = linear([xt,c],size,False)
      with tf.variable_scope("rnn_decoder",reuse=True):
          h_now,s_now = cell_decoder(xt_c,s_pre) # h_now = [batch_size, rnn_size]
      
      c,a = attention(s_now,True)
      with tf.variable_scope("AttnOutputProjection",reuse=True):
          output = linear(list(s_now)+[xt,c], size, True)

      y_logits = tf.matmul(output,w)+b #这里的h_now可以加上attention的vec?
      softmax = tf.nn.softmax(y_logits)
      log_prob = tf.log(softmax)
      next_token = tf.cast(tf.reshape(tf.multinomial(log_prob, 1), [batch_size]), tf.int32)
      gen_tokens_g = gen_tokens_g.write(i,next_token)
      x_tp1 = tf.nn.embedding_lookup(self.embedding, next_token)  # batch x emb_dim
      gen_as = gen_as.write(i,a)
      return i+1, x_tp1, s_now, gen_tokens_g, c , gen_as
    decoder_begin_token = self.input_decoder_pl[0]
    decoder_begin_token_embedding = tf.nn.embedding_lookup(self.embedding, decoder_begin_token)
    _,_, current_state, gen_tokens_g, atten, gen_as= control_flow_ops.while_loop(
              cond=lambda i, *_: i < self.buckets[-1][1],
              body=_generator_rnn_recurrence,
              loop_vars=( 0, decoder_begin_token_embedding, encoder_final_state, gen_tokens_g, atten, gen_as))
    
    self.gen_tokens_g = gen_tokens_g.stack() 
    self.gen_as = gen_as.stack()

    tf.summary.scalar('loss', self.loss)
    self.summary = tf.summary.merge_all()


  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.loss,  # Update Op that does SGD.
                     self.train_op]  # Loss for this batch.
    else:
      output_feed = [self.loss,self.summary]  # Loss for this batch.
    input_feed = {self.input_encoder_pl:encoder_inputs,
                  self.input_decoder_pl:decoder_inputs,
                  self.targets_decoder_pl:decoder_inputs[1:] + [decoder_inputs[-1]],
                  self.targets_weight_pl:target_weights
                  }
    outputs = session.run(output_feed, input_feed)
    return outputs[0],outputs[1]

  def chat_generate(self, session, encoder_inputs, decoder_start_tokens):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    output_feed = [self.gen_tokens_g,  # tokens
                     self.gen_as]  # attention weight
    input_feed = {self.input_encoder_pl:encoder_inputs, self.input_decoder_pl:decoder_start_tokens}
    tokens,gen_as = session.run(output_feed, input_feed)
    return tokens,gen_as

  def get_batch(self, data, bucket_id, batch_size=FLAGS.batch_size):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    over_sig = False
    for _ in xrange(batch_size):
      encoder_input = []
      decoder_input = []
      if self.bucket_index_d.has_key(bucket_id):
        self.bucket_index_d[bucket_id] += 1
        if self.bucket_index_d[bucket_id] >= len(data[bucket_id]):
          encoder_input, decoder_input = random.choice(data[bucket_id])
          #print("--pick bucket_id:%d random" %(bucket_id))
        else:
          over_sig = True
          encoder_input, decoder_input = data[bucket_id][self.bucket_index_d[bucket_id]]
          #print("--pick bucket_id:%d index:%d" %(bucket_id, self.bucket_index_d[bucket_id]))
      else:
        self.bucket_index_d[bucket_id] = 0
        encoder_input, decoder_input = data[bucket_id][self.bucket_index_d[bucket_id]]
        #print("--pick bucket_id:%d index:%d" %(bucket_id, self.bucket_index_d[bucket_id]))

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(batch_size, dtype=np.float32)
      for batch_idx in xrange(batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights, over_sig
