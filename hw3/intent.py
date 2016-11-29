# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 16:23:37 2016

@author: Bing Liu (liubing@cmu.edu)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import multi_task_model

import subprocess
import stat
# import shutil


tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 256, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("in_vocab_size", 10000, "max vocab Size.") 
tf.app.flags.DEFINE_integer("out_vocab_size", 10000, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "./data/", "Data directory")
# tf.app.flags.DEFINE_string("data_dir", "./tdata/ATIS_samples/", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./train/", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 250,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 7050,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 10000,
                            "Max size of test set.")
tf.app.flags.DEFINE_boolean("use_attention", True,
                            "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 47,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.5,
                          "dropout keep cell input and output prob.")  
tf.app.flags.DEFINE_boolean("bidirectional_rnn", True,
                            "Use birectional RNN")
tf.app.flags.DEFINE_string("task", "intent", "Options: joint; intent; tagging")
tf.app.flags.DEFINE_string("goal", "", "predict, train")
tf.app.flags.DEFINE_string("answer_path","","answer file path")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print ('Please indicate max sequence length. Exit')
    exit()

if FLAGS.task is None:
    print ('Please indicate task to run. Available options: intent; tagging; joint')
    exit()

task = dict({'intent':1, 'tagging':0, 'joint':0})

_buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]

def read_data(source_path, target_path, label_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      with tf.gfile.GFile(label_path, mode="r") as label_file:
        source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
        counter = 0
        while source and target and label and (not max_size or counter < max_size):
          counter += 1
          if counter % 100000 == 0:
            print("  reading data line %d" % counter)
            sys.stdout.flush()
          source_ids = [int(x) for x in source.split()]
          target_ids = [int(x) for x in target.split()]
          label_ids = [int(x) for x in label.split()]
#          target_ids.append(data_utils.EOS_ID)
          for bucket_id, (source_size, target_size) in enumerate(_buckets):
            if len(source_ids) < source_size and len(target_ids) < target_size:
              data_set[bucket_id].append([source_ids, target_ids, label_ids])
              break
          source, target, label = source_file.readline(), target_file.readline(), label_file.readline()
  return data_set # 3 outputs in each unit: source_ids, target_ids, label_ids 

def create_model(session, source_vocab_size, target_vocab_size, label_vocab_size):
  """Create model and initialize or load parameters in session."""
  with tf.variable_scope("model", reuse=None):
    model_train = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=False,
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)
  with tf.variable_scope("model", reuse=True):
    model_test = multi_task_model.MultiTaskModel(
          source_vocab_size, target_vocab_size, label_vocab_size, _buckets,
          FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
          dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
          forward_only=True,
          use_attention=FLAGS.use_attention,
          bidirectional_rnn=FLAGS.bidirectional_rnn,
          task=task)


  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model_train.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model_train, model_test

def train():
  print ('Applying Parameters:')
  for k,v in FLAGS.__dict__['__flags'].iteritems():
    print ('%s: %s' % (k, str(v)))
  print("Preparing data in %s" % FLAGS.data_dir)
  best_step = 0
  vocab_path = ''
  tag_vocab_path = ''
  label_vocab_path = ''
  in_seq_train, out_seq_train, label_train, in_seq_dev, out_seq_dev, label_dev, in_seq_test, out_seq_test, label_test, vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)

  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)

  with tf.Session() as sess:
    # Create model.
    print("Max sequence length: %d." % _buckets[0][0])
    print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

    model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab))
    print ("Creating model with source_vocab_size=%d, target_vocab_size=%d, and label_vocab_size=%d." % (len(vocab), len(tag_vocab), len(label_vocab)))

    # Read data into buckets and compute their sizes.
    print ("Reading train/valid/test data (training set limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(in_seq_dev, out_seq_dev, label_dev)
    # self.test_set = read_data(in_seq_test, out_seq_test, label_test)
    train_set = read_data(in_seq_train, out_seq_train, label_train)
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0

    best_valid_acuracy = 0
    best_valid_score = 0
    best_test_score = 0
    while model.global_step.eval() < FLAGS.max_training_steps:
      sys.stdout.write("\rglobal step %d" % (model.global_step.eval()))
      sys.stdout.flush()
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      # Get a batch and make a step.
      start_time = time.time()
      encoder_inputs, tags, tag_weights, batch_sequence_length, labels = model.get_batch(train_set, bucket_id)
      _, step_loss, classification_logits = model.classification_step(sess, encoder_inputs, labels,
                                   batch_sequence_length, bucket_id, False) 
      step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
      loss += step_loss / FLAGS.steps_per_checkpoint
      current_step += 1

      # Once in a while, we save checkpoint, print statistics, and run evals.
      if current_step % FLAGS.steps_per_checkpoint == 0:
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d step-time %.2f. Training perplexity %.2f" 
            % (model.global_step.eval(), step_time, perplexity))
        sys.stdout.flush()
        # Save checkpoint and zero timer and loss.
        checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0

        def run_valid_test(data_set, mode): # mode: Eval, Test
        # Run evals on development/test set and print the accuracy.
            word_list = list()
            ref_tag_list = list()
            hyp_tag_list = list()
            ref_label_list = list()
            hyp_label_list = list()
            correct_count = 0
            accuracy = 0.0
            tagging_eval_result = dict()
            for bucket_id in xrange(len(_buckets)):
              eval_loss = 0.0
              count = 0
              for i in xrange(len(data_set[bucket_id])):
                count += 1
                encoder_inputs, tags, tag_weights, sequence_length, labels = model_test.get_one(
                  data_set, bucket_id, i)
                tagging_logits = []
                classification_logits = []
                _, step_loss, classification_logits = model_test.classification_step(sess, encoder_inputs, labels,
                                             sequence_length, bucket_id, True) 
                eval_loss += step_loss / len(data_set[bucket_id])
                hyp_label = None
                ref_label_list.append(rev_label_vocab[labels[0][0]])
                hyp_label = np.argmax(classification_logits[0],0)
                hyp_label_list.append(rev_label_vocab[hyp_label])
                if labels[0] == hyp_label:
                  correct_count += 1

            accuracy = float(correct_count)*100/count
            print("\n accuracy: %.2f %d/%d" % (accuracy, correct_count, count))
            return accuracy, tagging_eval_result

        # valid
        valid_accuracy, valid_tagging_result = run_valid_test(dev_set, 'Eval')
        if valid_accuracy >= best_valid_acuracy:
            best_step = model.global_step.eval()
            best_valid_acuracy = valid_accuracy
  print(best_step)

def predict(): # mode: Eval, Test
# Run evals on development/test set and print the accuracy.
  _, _, _, _, _, _, in_seq_test, out_seq_test, label_test, vocab_path, tag_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
    FLAGS.data_dir, FLAGS.in_vocab_size, FLAGS.out_vocab_size)
  data_set = read_data(in_seq_test, out_seq_test, label_test)
  vocab, rev_vocab = data_utils.initialize_vocabulary(vocab_path)
  tag_vocab, rev_tag_vocab = data_utils.initialize_vocabulary(tag_vocab_path)
  label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
  mode = "Test"
  with tf.Session() as sess:
    model, model_test = create_model(sess, len(vocab), len(tag_vocab), len(label_vocab))
    word_list = list()
    ref_tag_list = list()
    hyp_tag_list = list()
    ref_label_list = list()
    hyp_label_list = list()
    correct_count = 0
    accuracy = 0.0
    tagging_eval_result = dict()
    for bucket_id in xrange(len(_buckets)):
      eval_loss = 0.0
      count = 0
      for i in xrange(len(data_set[bucket_id])):
        count += 1
        encoder_inputs, tags, tag_weights, sequence_length, labels = model_test.get_one(
          data_set, bucket_id, i)
        tagging_logits = []
        classification_logits = []
        _, step_loss, classification_logits = model_test.classification_step(sess, encoder_inputs, labels,
                                     sequence_length, bucket_id, True) 
        hyp_label = None
        hyp_label = np.argmax(classification_logits[0],0)
        hyp_label_list.append(rev_label_vocab[hyp_label])
        with open(FLAGS.answer_path,'w') as output_f:
          output_f.write('\n'.join(hyp_label_list))

def main(_):
    if FLAGS.goal == "train":
        train()
    elif FLAGS.goal == "predict": 
        predict()

if __name__ == "__main__":
  tf.app.run()


