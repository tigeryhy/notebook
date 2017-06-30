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


def train():
    print("Preparing dialog data in %s" % FLAGS.data_dir)
    train_data, dev_data, _ = data_utils.prepare_dialog_data(FLAGS.data_dir, FLAGS.vocab_size)

    with tf.Session() as sess:

        # Create model.
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model,current_step = create_model(sess, forward_only=False)

        # Read data into buckets and compute their sizes.
        print ("Reading development and training data (limit: %d)." % FLAGS.max_train_data_size)
        dev_set = read_data(dev_data)
        train_set = read_data(train_data, FLAGS.max_train_data_size)
        train_bucket_sizes = [len(train_set[b]) for b in xrange(len(BUCKETS))]
        train_total_size = float(sum(train_bucket_sizes))

        # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
        # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
        # the size if i-th training bucket, as used later.
        train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                               for i in xrange(len(train_bucket_sizes))]
        print "--train_bucket_sizes:",train_bucket_sizes
        print "--train_total_size:",train_total_size
        print "--train_buckets_scale:",train_buckets_scale

        summary_writer = tf.summary.FileWriter(FLAGS.model_dir)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        previous_losses = []
        time_beg = 0
        time_end = 0

        while True:
          # Choose a bucket according to data distribution. We pick a random number
          # in [0, 1] and use the corresponding interval in train_buckets_scale.
          random_number_01 = np.random.random_sample()
          bucket_id = min([i for i in xrange(len(train_buckets_scale))
                           if train_buckets_scale[i] > random_number_01])

          # Get a batch and make a step.
          start_time = time.time()
          encoder_inputs, decoder_inputs, target_weights, over_sig = model.get_batch(
              train_set, bucket_id)

          step_loss,_ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, forward_only=False)

          step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
          loss += step_loss / FLAGS.steps_per_checkpoint
          current_step += 1
          if current_step % 100 == 0:
              print("%d step : flex :%.4f" %(current_step,np.exp(step_loss)))

          # Once in a while, we save checkpoint, print statistics, and run evals.
          if current_step % FLAGS.steps_per_checkpoint == 1:
            if time_beg != 0:
              print ("100 step time: %.2f" %(start_time - time_beg))
            time_beg = start_time

            previous_losses.append(loss)

            # Save checkpoint and zero timer and loss.
            checkpoint_path = os.path.join(FLAGS.model_dir, "model.ckpt")
            model.saver.save(sess, checkpoint_path, global_step=current_step)
            step_time, loss = 0.0, 0.0

            # Run evals on development set and print their perplexity.
           
            for bucket_id in xrange(len(BUCKETS)):
              encoder_inputs, decoder_inputs, target_weights, _ = model.get_batch(dev_set, bucket_id)
              eval_loss,summary_str = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)

              if bucket_id == 0:
                  summary_writer.add_summary(summary_str,current_step)
                  summary_writer.flush()
              eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
              print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
            

            print("3 round loss diff: %.2f" % (max(previous_losses[-3:]) - min(previous_losses[-3:])))
            sys.stdout.flush()
            if over_sig and len(previous_losses) > 2 and max(previous_losses[-3:]) - min(previous_losses[-3:]) < 0.01:
              break
def main(_):
    train()

if __name__ == "__main__":
    tf.app.run()
