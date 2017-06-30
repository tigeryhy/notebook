from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

from config import FLAGS, BUCKETS
import data_utils
import seq2seq_model


def create_model(session, forward_only):
  """Create translation model and initialize or load parameters in session."""
  model = seq2seq_model.Seq2SeqModel(
      vocab_size=FLAGS.vocab_size,
      buckets=BUCKETS,
      size=FLAGS.size,
      num_layers=FLAGS.num_layers,
      max_gradient_norm=FLAGS.max_gradient_norm,
      batch_size=FLAGS.batch_size,
      forward_only=forward_only)

  ckpt = tf.train.latest_checkpoint(FLAGS.model_dir)
  print(ckpt)
  start_epoch = 0
  if ckpt :
    print("Reading model parameters from %s" % ckpt)
    model.saver.restore(session, ckpt)
    print("[INFO] restore from the checkpoint {0}".format(ckpt))
    start_epoch += int(ckpt.split('-')[-1])
  else:
    print("Created model with fresh parameters.")
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    session.run(init_op)
  return model,start_epoch

