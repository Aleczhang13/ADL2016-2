import sys
import os
import numpy as np
import math
import time
import itertools
import shutil
import tensorflow as tf
import tree as tr
from tqdm import tqdm
import sys


RESET_AFTER = 50
class Config(object):
    embed_size = 128
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 10
    lr = 0.01
    l2 = 0.02
    model_name ='model_rnn'


class RNN_Model():
    def load_embedding(self):
        w2v={}
        with open('w2v.txt','rt') as f:
            for line in f.readlines():
                x = line.strip().split(' ')
                word = x[0]
                embedding = [float(i) for i in x[1:]]
                w2v[word]=embedding
        return w2v

    def load_data(self):
        self.test_data = tr.load_test_data()
        self.embeddings = self.load_embedding()

    def get_embedding(self, word):
        embedding = None
        if word in self.embeddings:
            embedding = self.embeddings[word]
        else:
            embedding = self.embeddings['UNK']
        return tf.constant(embedding)


    def inference(self, tree):
        logits = self.add_projections(self.add_model(tree.root))
        return logits

    def add_model_vars(self):
        with tf.variable_scope('Composition'):
            tf.get_variable('W1', [2 * self.config.embed_size, self.config.embed_size])
            tf.get_variable('b1', [1, self.config.embed_size])
        with tf.variable_scope('Projection'):
            tf.get_variable('U', [self.config.embed_size, self.config.label_size])

    def add_model(self, node):
        with tf.variable_scope('Composition', reuse=True):
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')
        if node.isLeaf:
            return self.get_embedding(node.word)
        else:
            left = self.add_model(node.left)
            right = self.add_model(node.right)
            lv = tf.reshape(left,[1,self.config.embed_size])
            rv = tf.reshape(right,[1,self.config.embed_size])
            child = tf.concat(0,[lv,rv])
            child = tf.reshape(child,[1,self.config.embed_size*2])
            tensor = tf.nn.tanh(tf.matmul(child,W1)+b1)
            return tensor

    def add_projections(self, node_tensors):
        logits = None
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
            logits = tf.matmul(node_tensors, U)
        return logits

    def loss(self, logits, labels):
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits,labels)
        return cross_entropy

    def training(self, loss):
        train_op = None
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        return train_op

    def predictions(self, y):
        predictions = None
        predictions = tf.argmax(y, 1)
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path):
        preds=[]
        step = 0
        for i in tqdm(xrange(int(math.ceil(len(trees)/float(RESET_AFTER))))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree)
                    preds.append(int(sess.run(self.predictions(logits))))
        return preds

def test_RNN():
    config = Config()
    model = RNN_Model(config)
    start_time = time.time()
    predictions= model.predict(model.test_data, './weights/%s'%model.config.model_name)
    with open(sys.argv[1],'w') as out_f:
        for pred in predictions:
            out_f.write(str(pred)+'\n')

if __name__ == "__main__":
        test_RNN()
