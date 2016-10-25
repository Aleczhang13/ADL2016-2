import tensorflow as tf
import numpy as np
from tqdm import tqdm
import collections
from collections import Counter, defaultdict
from scipy import sparse
from random import shuffle
import sys
import getopt


input_file_name = ''
output_file_name = ''
option, _ = getopt.getopt(sys.argv[1:], 'i:o:', ['-i','-o'])
for opt, arg in option:
    if opt == '-i':
        input_file_name = arg
    elif opt == '-o':
        output_file_name = arg


def read_data(f_name):
    print("reading data")
    with open(f_name, "r") as f:
        words = tf.compat.as_str(f.read())
    return words
corpus = read_data(input_file_name).split()

def context_windows(region, left_size, right_size):
    for i, word in enumerate(region):
        start_index = i - left_size
        end_index = i + right_size
        left_context = window(region, start_index, i - 1)
        right_context = window(region, i + 1, end_index)
        yield (left_context, word, right_context)

def window(region, start_index, end_index):
    last_index = len(region) + 1
    selected_tokens = region[max(start_index, 0):min(end_index, last_index) + 1]
    return selected_tokens

def fit_to_corpus(corpus, vocab_size, min_occurrences, left_size, right_size):
    print('fit to corpus')
    word_counts = Counter()
    cooccurrence_counts = defaultdict(float)
    for region in [corpus]:
        word_counts.update(region)
        for l_context, word, r_context in context_windows(region, left_size, right_size):
            for i, context_word in enumerate(l_context[::-1]):
                # add (1 / distance from focal word) for this pair
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
            for i, context_word in enumerate(r_context):
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
    if len(cooccurrence_counts) == 0:
        raise ValueError("No coccurrences in corpus. Did you try to reuse a generator?")
    words = [word for word, count in word_counts.most_common(vocab_size)
                    if count >= min_occurrences]
    word_to_id = {word: i for i, word in enumerate(words)}
    cooccurrence_matrix = {
        (word_to_id[words[0]], word_to_id[words[1]]): count
        for words, count in cooccurrence_counts.items()
        if words[0] in word_to_id and words[1] in word_to_id}
    return words, word_to_id, cooccurrence_matrix

_, dictionary, cooccurrence_matrix = fit_to_corpus(corpus,100000,1,5,5)
reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
voc_size = len(dictionary)

embedding_size = 200
batch_size = 500



graph = tf.Graph()
with graph.as_default():
    print('build graph')
    count_max = tf.constant([100], dtype=tf.float32)
    scaling_factor = tf.constant([3.0/4.0], dtype=tf.float32)
    focal_input = tf.placeholder(tf.int32, shape=[batch_size])
    context_input = tf.placeholder(tf.int32, shape=[batch_size])
    cooccurrence_count = tf.placeholder(tf.float32, shape=[batch_size])
    focal_embeddings = tf.Variable(
        tf.random_uniform([voc_size, embedding_size], 1.0, -1.0))
    context_embeddings = tf.Variable(
        tf.random_uniform([voc_size, embedding_size], 1.0, -1.0))
    focal_biases = tf.Variable(tf.random_uniform([voc_size], 1.0, -1.0))
    context_biases = tf.Variable(tf.random_uniform([voc_size], 1.0, -1.0))
    focal_embedding = tf.nn.embedding_lookup([focal_embeddings], focal_input)
    context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
    focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
    context_bias = tf.nn.embedding_lookup([context_biases], context_input)

    weighting_factor = tf.minimum(
        1.0,
        tf.pow(
            tf.div(cooccurrence_count, count_max),
            scaling_factor))

    embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1)
    log_cooccurrences = tf.log(tf.to_float(cooccurrence_count))

    distance_expr = tf.square(tf.add_n([
        embedding_product,
        focal_bias,
        context_bias,
        tf.neg(log_cooccurrences)]))

    single_losses = tf.mul(weighting_factor, distance_expr)
    total_loss = tf.reduce_sum(single_losses)
    optimizer = tf.train.AdagradOptimizer(0.05).minimize(total_loss)
    summary = tf.merge_all_summaries()
    combined_embeddings = tf.add(focal_embeddings, context_embeddings)



def batchify(batch_size, *sequences):
    for i in range(0, len(sequences[0]), batch_size):
        yield tuple(sequence[i:i+batch_size] for sequence in sequences)

def prepare_batches():
    print('prepare batches')
    cooccurrences = [(word_ids[0], word_ids[1], count)
                     for word_ids, count in cooccurrence_matrix.items()]
    i_indices, j_indices, counts = zip(*cooccurrences)
    return list(batchify(batch_size, i_indices, j_indices, counts))

batches = prepare_batches()


def write_file(name, embeddings):
    output = open(name,'w')
    print('write file')
    for i in range(voc_size):
        output.write(reverse_dictionary[i]+' ')
        for j in range(embedding_size):
            output.write(str(embeddings[i][j])+' ')
        output.write('\n')
    output.close


print("start train")
embeddings = ''
num_epochs = 1
with tf.Session(graph=graph) as session:
    global embeddings
    tf.initialize_all_variables().run()
    for epoch in tqdm(range(num_epochs)):
        shuffle(batches)
        for batch_index in range(len(batches)):
            batch = batches[batch_index]
            i_s, j_s, counts = batch
            if len(counts) != batch_size:
                continue
            feed_dict = {
                focal_input: i_s,
                context_input: j_s,
                cooccurrence_count: counts}
            session.run(optimizer, feed_dict=feed_dict)
    embeddings = combined_embeddings.eval()
    print('write file')
    write_file(output_file_name, embeddings)

