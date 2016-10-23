import tensorflow as tf
import numpy as np
from tqdm import tqdm
import collections
from scipy import sparse

# input_file_name = "text8"
input_file_name = "test.txt"

def read_data(f_name):
    print("reading data")
    with open(f_name, "r") as f:
        words = tf.compat.as_str(f.read()).split()
    return words

words = read_data(input_file_name)


def build_dict(words):
    print("build dict")
    count = collections.Counter(words).most_common()
    dictionary = {}
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(),dictionary.keys()))
    data = list()
    for word in words:
        if word in dictionary:
            data.append(dictionary[word])
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dict(words)
voc_size = len(dictionary)
data_size = len(data)-1

def build_dataset(context_size):
    context_size = 1
    global data, dictionary, reverse_dictionary
    X = sparse.lil_matrix((voc_size, voc_size),dtype=np.float64)
    for pos, w_id in enumerate(data):
        for i in range(1, min(context_size, data_size-pos)+1):
            increment = 1 # may be affect by distance
            X[w_id, data[pos+i]] += 1
            X[data[pos+i], w_id] += 1


# Step 3: Function to generate a training batch for the skip-gram model.
def generate_batch(batch_size, num_skips, skip_window):
  global data_index
  assert batch_size % num_skips == 0
  assert num_skips <= 2 * skip_window
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  span = 2 * skip_window + 1 # [ skip_window target skip_window ]
  buffer = collections.deque(maxlen=span)
  for _ in range(span):
    buffer.append(data[data_index])
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
