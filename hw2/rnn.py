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
from utils import Vocab
from gensim.models import Word2Vec


RESET_AFTER = 50
class Config(object):
    embed_size = 25
    label_size = 2
    early_stopping = 2
    anneal_threshold = 0.99
    anneal_by = 1.5
    max_epochs = 15
    lr = 0.15
    l2 = 0.02
    model_name = 'rnn_embed=%d_l2=%f_lr=%f.weights'%(embed_size, l2, lr)


class RNN_Model():

    def load_data(self):
        # self.train_data, self.test_data = tr.simplified_data('data/test_tree_pos','data/test_tree_neg','data/test_tree_pos')
        self.train_data, self.test_data = tr.simplified_data('data/bin_tree.pos','data/bin_tree.neg','data/bin_tree.test')

        # build vocab from training data
        self.w2v_model = Word2Vec.load_word2vec_format('data/w2v_vec')
        dictionary = {}
        embeddings = None
        for word in self.w2v_model.vocab:
            if len(dictionary) == 0:
                embeddings = self.w2v_model['word'].reshape(1,len(self.w2v_model['word']))
            else:
                embeddings = np.vstack([embeddings,self.w2v_model[word]])
            dictionary[word] = len(dictionary)
        re_dictionary = {index : word for word, index in dictionary.items()}
        self.vocab = Vocab()
        self.vocab.construct(re_dictionary, dictionary)
        self.embeddings = embeddings

    def inference(self, tree):
        """For a given tree build the RNN models computation graph up to where it
            may be used for inference.
        Args:
            tree: a Tree object on which to build the computation graph for the RNN
        Returns:
            softmax_linear: Output tensor with the computed logits.
        """
        node_tensors = self.add_model(tree.root)
        node_tensors = node_tensors[tree.root]
        return self.add_projections(node_tensors)

    def add_model_vars(self):
        '''
        You model contains the following parameters:
            embedding:  tensor(vocab_size, embed_size)
            W1:         tensor(2* embed_size, embed_size)
            b1:         tensor(1, embed_size)
            U:          tensor(embed_size, output_size)
            bs:         tensor(1, output_size)
        Hint: Add the tensorflow variables to the graph here and *reuse* them while building
                the compution graphs for composition and projection for each tree
        Hint: Use a variable_scope "Composition" for the composition layer, and
              "Projection") for the linear transformations preceding the softmax.
        '''
        with tf.variable_scope('Composition'):
            tf.get_variable('embedding', initializer=tf.constant(self.embeddings), trainable= False)
            tf.constant(self.embeddings,name='embedding')
            tf.get_variable('W1', [2 * self.config.embed_size, self.config.embed_size])
            tf.get_variable('b1', [1, self.config.embed_size])
        with tf.variable_scope('Projection'):
            tf.get_variable('U', [self.config.embed_size, 2])
            # tf.get_variable('bs', [1, ])

    def add_model(self, node):
        """Recursively build the model to compute the phrase embeddings in the tree

        Hint: Refer to tree.py and vocab.py before you start. Refer to
              the model's vocab with self.vocab
        Hint: Reuse the "Composition" variable_scope here
        Hint: Store a node's vector representation in node.tensor so it can be
              used by it's parent
        Hint: If node is a leaf node, it's vector representation is just that of the
              word vector (see tf.gather()).
        Args:
            node: a Node object
        Returns:
            node_tensors: Dict: key = Node, value = tensor(1, embed_size)
        """
        with tf.variable_scope('Composition', reuse=True):
            embedding = tf.get_variable('embedding')
            W1 = tf.get_variable('W1')
            b1 = tf.get_variable('b1')


        node_tensors = dict()  # {node: tensor}
        curr_node_tensor = None
        if node.isLeaf:
            word_id = self.vocab.encode(node.word)
            #return embedding 
            curr_node_tensor = tf.expand_dims(tf.gather(embedding, word_id), 0)
        else:
            # not leaf, recursive
            # update(), append object on dict
            node_tensors.update(self.add_model(node.left))
            node_tensors.update(self.add_model(node.right))
            child_tensor = tf.concat(1, [node_tensors[node.left], node_tensors[node.right]])
            # curr_node_tensor = tf.nn.relu(tf.matmul(child_tensor, W1) + b1)
            curr_node_tensor = tf.nn.tanh(tf.matmul(child_tensor, W1) + b1)
        node_tensors[node] = curr_node_tensor
        return node_tensors

    def add_projections(self, node_tensors):
        """Add projections to the composition vectors to compute the raw sentiment scores

        Hint: Reuse the "Projection" variable_scope here
        Args:
            node_tensors: tensor(?, embed_size)
        Returns:
            output: tensor(?, label_size)
        """
        logits = None
        with tf.variable_scope('Projection', reuse=True):
            U = tf.get_variable('U')
            # bs = tf.get_variable('bs')
            logits = tf.matmul(node_tensors, U)
            # logits += bs
        return logits

    def loss(self, logits, labels):
        """Adds loss ops to the computational graph.

        Hint: Use sparse_softmax_cross_entropy_with_logits
        Hint: Remember to add l2_loss (see tf.nn.l2_loss)
        Args:
            logits: tensor(num_nodes, output_size)
            labels: python list, len = num_nodes
        Returns:
            loss: tensor 0-D
        """
        loss = None
        # YOUR CODE HERE
        # with tf.variable_scope('Composition', reuse=True):
        #     W1 = tf.get_variable('W1')
        # with tf.variable_scope('Projection', reuse=True):
        #     U = tf.get_variable('U')
        # l2loss = tf.nn.l2_loss(W1) + tf.nn.l2_loss(U)
        cross_entropy = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels))
        loss = cross_entropy
        # loss = cross_entropy + self.config.l2 * l2loss
        # END YOUR CODE
        return loss

    def training(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Hint: Use tf.train.GradientDescentOptimizer for this model.
                Calling optimizer.minimize() will return a train_op object.

        Args:
            loss: tensor 0-D
        Returns:
            train_op: tensorflow op for training.
        """
        train_op = None
        # YOUR CODE HERE
        optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
        train_op = optimizer.minimize(loss)
        # END YOUR CODE
        return train_op

    def predictions(self, y):
        """Returns predictions from sparse scores

        Args:
            y: tensor(?, label_size)
        Returns:
            predictions: tensor(?,1)
        """
        predictions = None
        # YOUR CODE HERE
        predictions = tf.argmax(y, 1)
        # END YOUR CODE
        return predictions

    def __init__(self, config):
        self.config = config
        self.load_data()

    def predict(self, trees, weights_path, get_loss = False):
        """Make predictions from the provided model."""
        results = []
        for i in xrange(int(math.ceil(len(trees)/float(RESET_AFTER)))):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                saver = tf.train.Saver()
                saver.restore(sess, weights_path)
                for tree in trees[i*RESET_AFTER: (i+1)*RESET_AFTER]:
                    logits = self.inference(tree)
                    predictions = self.predictions(logits)
                    root_prediction = sess.run(predictions)[0]
                    results.append(root_prediction)
        return results

    def run_epoch(self,epoch, new_model = False, verbose=True):
        step = 0
        t1 = time.time()
        while step < len(self.train_data):
            with tf.Graph().as_default(), tf.Session() as sess:
                self.add_model_vars()
                if new_model:
                    init = tf.initialize_all_variables()
                    sess.run(init)
                else:
                    saver = tf.train.Saver()
                    saver.restore(sess, './weights/%s'%self.config.model_name)
                t2 = time.time()
                for _ in xrange(RESET_AFTER):
                    if step >= len(self.train_data):
                        break
                    tree = self.train_data[step]
                    logits = self.inference(tree) # tree embeddings
                    labels = [tree.label]
                    loss = self.loss(logits, labels)
                    train_op = self.training(loss)
                    loss, _ = sess.run([loss, train_op])
                    step += 1
                saver = tf.train.Saver()
                if not os.path.exists("./weights"):
                    os.makedirs("./weights")
                saver.save(sess, './weights/%s'%self.config.model_name)
                t3 = time.time()
                sys.stdout.write('\repoch: %d/%d, step: %d/%d, total_time: %d sec, stage time: %d sec' %(epoch,self.config.max_epochs, step,len(self.train_data), t3-t1, t3-t2))
                sys.stdout.flush()

    def train(self, verbose=True):
        stopped = -1
        for epoch in xrange(self.config.max_epochs):
            start_time = time.time()
            if epoch==0:
                self.run_epoch(epoch, new_model=True)
            else:
                self.run_epoch(epoch)
            print 'Training time: {}'.format(time.time()- start_time)


def test_RNN():
    """Test RNN model implementation.

    You can use this function to test your implementation of the Named Entity
    Recognition network. When debugging, set max_epochs in the Config object to 1
    so you can rapidly iterate.
    """
    print "set up config"
    config = Config()
    print "set up model"
    model = RNN_Model(config)
    print "start train"
    stats = model.train(verbose=True)
    predictions = model.predict(model.test_data, './weights/%s'%model.config.model_name)
    f = open('output.txt','w')
    for pred in predictions:
        f.write(str(pred)+'\n')
    f.close()

    # labels = [t.root.label for t in model.test_data]
    # test_acc = np.equal(predictions, labels).mean()
    # print 'Test acc: {}'.format(test_acc)

if __name__ == "__main__":
        test_RNN()
