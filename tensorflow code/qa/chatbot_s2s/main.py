import sys
import os
import math
import time

import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf

from data_utils import read_data
import data_utils
import seq2seq_model
from seq2seq_model_utils import create_model
from config import FLAGS,BUCKETS
from train import train
from predict import predict


def main(_):
    if FLAGS.type == "train":
        train()
    elif FLAGS.type == "predict":
        query = FLAGS.query
        predict(query)

if __name__ == "__main__":
    tf.app.run()
