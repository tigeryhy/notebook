# -*- coding: utf-8 -*-
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


def predict(query):
    print("Preparing dialog data in %s" % FLAGS.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:
        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model,current_step = create_model(sess, forward_only=False)
        vocab_path = os.path.join(FLAGS.data_dir, "vocab%d.in" % FLAGS.vocab_size)
        vocab, reverse_vocab = data_utils.initialize_vocabulary(vocab_path)

        query_ids = [vocab[word] for word in query.split("__")]
        query_ids = list(reversed(query_ids)) + [data_utils.PAD_ID]*3
        query_ids = np.array(query_ids).reshape(-1,1)
        print("query_ids : %s", query_ids)
        decoder_start_tokens = [[data_utils.GO_ID]]

        tokens,gen_as = model.chat_generate(sess, query_ids,decoder_start_tokens)
        token_ids  = tokens.reshape([-1]).tolist()
        answer = " ".join([reverse_vocab[token_id] for token_id in token_ids])
        print(answer)
        print(gen_as)
def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
