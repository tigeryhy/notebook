# -*- coding: utf-8 -*-
import tensorflow as tf

TEST_DATASET_PATH = 'data/qa.test.set'
SAVE_DATA_DIR = 'save_dir/'

tf.app.flags.DEFINE_string('type', "train", 'train or predict')
tf.app.flags.DEFINE_string('query', "", 'predict\'s query')

tf.app.flags.DEFINE_string('data_dir', SAVE_DATA_DIR + 'data', 'Data directory')
tf.app.flags.DEFINE_string('model_dir', SAVE_DATA_DIR + 'nn_models', 'Train directory')
tf.app.flags.DEFINE_string('results_dir', SAVE_DATA_DIR + 'results', 'Train directory')

tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 256, 'Batch size to use during training.')

tf.app.flags.DEFINE_integer('vocab_size', 500000, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('size', 512, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', 2, 'Number of layers in the model.')

tf.app.flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 1000, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
BUCKETS = [(5, 10), (10, 15), (20, 25),(30,40), (40, 50)]
#BUCKETS = [(5, 10), (10, 15), (20, 25)]



