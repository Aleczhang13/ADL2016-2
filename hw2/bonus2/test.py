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
import sys
# Parameters
# ==================================================

input_path = sys.argv[1]
output_path = sys.argv[2]
#TODO bash input
tf.flags.DEFINE_string("test_data_file", input_path, "Data source for the positive data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
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
print "load w2v model"
w2v_model = Word2Vec.load_word2vec_format('./w2v_vec_128')
dictionary = {'<unk>': 0}
embeddings = None
# unk = np.random.rand(128).astype(np.float32)
# unk = normalize(unk[:,np.newaxis], axis=0).ravel()
unk = np.array([ 0.1143818 ,0.00717914 ,0.04847993 ,0.0910058  ,0.02960362 , 0.00079644
                 ,0.01670034  ,0.02926759  ,0.00486377  ,0.11733559 , 0.15520912 , 0.05046108
                 ,0.0779002   ,0.14043896  ,0.09469795  ,0.10766723 , 0.02045983 , 0.1571952
                 ,0.08316572  ,0.12017845  ,0.08433241  ,0.09749743 , 0.0139061  , 0.00736295
                 ,0.11027702  ,0.10882029  ,0.05822834  ,0.09762562 , 0.04472737 , 0.029959
                 ,0.1404493   ,0.15723996  ,0.1353804   ,0.12785764 , 0.01740257 , 0.07184425
                 ,0.12085188  ,0.04723262  ,0.04522007  ,0.10222704 , 0.04699707 , 0.03832452
                 ,0.02217182  ,0.02115907  ,0.15985888  ,0.01842072 , 0.1475596  , 0.02647784
                 ,0.03574994  ,0.13515423  ,0.08122846  ,0.07374505 , 0.08650312 , 0.03177773
                 ,0.01887928  ,0.13036889  ,0.06443241  ,0.13868879 , 0.15305461 , 0.11800312
                 ,0.15540311  ,0.09540048  ,0.02991556  ,0.11723517 , 0.01983726 , 0.09771425
                 ,0.15015458  ,0.07044076  ,0.14816664  ,0.04594498 , 0.05400189 , 0.09718105
                 ,0.01280875  ,0.08384718  ,0.10292216  ,0.10032745 , 0.0905977  , 0.05968845
                 ,0.05465133  ,0.05867483  ,0.119304    ,0.09425836 , 0.05671079 , 0.020384
                 ,0.12151096  ,0.01297562  ,0.11297181  ,0.1137782  , 0.14710623 , 0.03227491
                 ,0.12615812  ,0.04006816  ,0.0396733   ,0.0481496  , 0.13155933 , 0.07610495
                 ,0.0278303   ,0.13273373  ,0.05803545  ,0.07918923 , 0.00771133 , 0.04416436
                 ,0.00054184  ,0.10236327  ,0.04904902  ,0.07428808 , 0.03696999 , 0.15593921
                 ,0.00840859  ,0.13573852  ,0.05039854  ,0.0593581  , 0.02655525 , 0.02537905
                 ,0.04800728  ,0.02752387  ,0.13295586  ,0.01705544 , 0.11676213 , 0.08950207
                 ,0.0399991   ,0.08891759  ,0.08498032  ,0.06310595 , 0.01851667 , 0.14893569
                 ,0.033753    ,0.09851403],dtype=np.float32)
embeddings = unk.reshape(1,len(unk))
for word in tqdm(w2v_model.vocab):
    embeddings = np.vstack([embeddings,w2v_model[word]])
    dictionary[word] = len(dictionary)
re_dictionary = {index : word for word, index in dictionary.items()}
vocab = Vocab()
vocab.construct(re_dictionary, dictionary)

# testing # Load data
print("Loading data...")
# TODO change train file name
max_document_length = 28
# Randomly shuffle data

raw_test_x = data_helpers.load_test_data(FLAGS.test_data_file)
# Build vocabulary
test_x = np.zeros([len(raw_test_x),max_document_length],dtype= np.int32)
for i, line in enumerate(raw_test_x):
    for j, word in enumerate(line.split()):
        test_x[i][j] = vocab.encode(word)

# Training
# ==================================================

graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        saver = tf.train.import_meta_graph("./weights/model.meta")
        saver.restore(sess,"./weights/model")
        # Define Training procedure
        # Get the placeholders from the graph by name

        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Collect the predictions here
        all_predictions = []

        predictions = sess.run(predictions, {input_x: test_x, dropout_keep_prob: 1.0})

        print "predicting"
        out_f = open(output_path,"w")
        for pred in predictions:
            out_f.write(str(pred)+'\n')
        out_f.close()
