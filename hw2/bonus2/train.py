#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from utils import Vocab
from gensim.models import Word2Vec
from sklearn.preprocessing import normalize
from tqdm import tqdm
from tensorflow.contrib import learn
from utils import Vocab
# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
# tf.flags.DEFINE_string("positive_data_file", "./data1/t_train_pos", "Data source for the positive data.")
tf.flags.DEFINE_string("positive_data_file", "./data1/training_data.pos", "Data source for the positive data.")
# tf.flags.DEFINE_string("negative_data_file", "./data1/t_train_neg", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data1/training_data.neg", "Data source for the positive data.")
tf.flags.DEFINE_string("test_data_file", "./data1/testing_data.txt", "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


# Data Preparatopn
# ==================================================
# build dictionary
w2v_model = Word2Vec.load_word2vec_format('w2v_vec_128')
dictionary = {'<unk>': 0}
embeddings = None
unk = np.random.rand(128).astype(np.float32)
unk = normalize(unk[:,np.newaxis], axis=0).ravel()
embeddings = unk.reshape(1,len(unk))
for word in tqdm(w2v_model.vocab):
    embeddings = np.vstack([embeddings,w2v_model[word]])
    dictionary[word] = len(dictionary)
re_dictionary = {index : word for word, index in dictionary.items()}
vocab = Vocab()
vocab.construct(re_dictionary, dictionary)

# testing
# Load data
print("Loading data...")
# TODO change train file name
raw_train_x, raw_train_y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
max_document_length = max([len(x.split(" ")) for x in raw_train_x])
# Build vocabulary
x = np.zeros([len(raw_train_x),max_document_length],dtype= np.int32)
for i, line in enumerate(raw_train_x):
    for j, word in enumerate(line.split()):
        x[i][j] = vocab.encode(word)

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(raw_train_y)))
x_train = x[shuffle_indices]
y_train = raw_train_y[shuffle_indices]

raw_test_x = data_helpers.load_test_data(FLAGS.test_data_file)

# Build test
test_x = np.zeros([len(raw_test_x),max_document_length],dtype= np.int32)
for i, line in enumerate(raw_test_x):
    for j, word in enumerate(line.split()):
        test_x[i][j] = vocab.encode(word)

# Training
# ==================================================

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        cnn = TextCNN(
            sequence_length=x_train.shape[1],
            num_classes=y_train.shape[1],
            vocab_size=len(dictionary),
            embedding_size=FLAGS.embedding_dim,
            filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
            num_filters=FLAGS.num_filters,
            embeddings=embeddings,
            l2_reg_lambda=FLAGS.l2_reg_lambda)

        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        if not os.path.exists('./weights'):
            os.makedirs('./weights')
        saver = tf.train.Saver(tf.all_variables())
        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            """
            A single training step
            """
            feed_dict = {
              cnn.input_x: x_batch,
              cnn.input_y: y_batch,
              cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }
            _ = sess.run(
                [train_op],
                feed_dict)

        # Generate batches
        batches = data_helpers.batch_iter(
            list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
        # Training loop. For each batch...
        for batch in tqdm(batches):
            x_batch, y_batch = zip(*batch)
            train_step(x_batch, y_batch)
        path = saver.save(sess, "./weights/model")
        # out_f = open("output.txt","w")
        # print "predictions"
        # predictions = sess.run(cnn.predictions, {cnn.input_x: test_x, cnn.dropout_keep_prob: 1.0})
        # for pred in predictions:
        #     out_f.write(str(pred)+'\n')
        # out_f.close()
