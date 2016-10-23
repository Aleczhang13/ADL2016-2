import tensorflow as tf
import numpy as np
from tqdm import tqdm
import collections
import random
import math

input_file_name = "text8"

def read_data(f_name):
    print("reading data")
    with open(f_name, "r") as f:
        words = tf.compat.as_str(f.read()).split()
    return words

words = read_data(input_file_name)

def build_dataset(words):
    print("build dataset")
    count = collections.Counter(words).most_common()
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    data = list()
    for word in words:
        data.append(dictionary[word])
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
del words
vocabulary_size = len(dictionary)

data_index = 0
def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    # batch shape 1xn
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # labels shape nx1
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    # +1 for range loop
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    # double-sided queue
    buffer = collections.deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        # append near by data [window target window]
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels



batch_size = 128
embedding_size = 128 # Dimension of the embedding vector.
skip_window = 1       # How many words to consider left and right.
num_skips = 2         # How many times to reuse an input to generate a label.


graph = tf.Graph()

with graph.as_default():

    # Input data.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
                            stddev=1.0 / math.sqrt(embedding_size)))


    loss = tf.nn.softmax(tf.matmul(embed, tf.transpose(weights)))

    # Construct the SGD optimizer using a learning rate of 1.0.
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm

    # Add variable initializer.
    init = tf.initialize_all_variables()

  # Step 5: Begin training.
num_steps = 100001

with tf.Session(graph=graph) as session:
    # We must initialize all variables before we use them.
    init.run()
    print("Initialized")

    print("start training")
    for step in tqdm(range(num_steps)):
        batch_inputs, batch_labels = generate_batch(
            batch_size, num_skips, skip_window)
        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}

        session.run([optimizer, loss], feed_dict=feed_dict)
    final_embeddings = normalized_embeddings.eval()
    f = open("output","w")
    row_num = 0
    print("writing output file")
    for row in final_embeddings:
        f.write(reverse_dictionary[row_num]+' ')
        for ele in row:
            f.write(str(ele)+' ')
        row_num += 1
        f.write("\n")
    f.close()
