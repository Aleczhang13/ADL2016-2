from collections import defaultdict


class Vocab(object):
    def __init__(self):
        self.word_to_index = {}
        self.index_to_word = {}
        self.unknown = '<unk>'

    def construct(self, index_to_word, word_to_index):
        self.word_to_index = word_to_index
        self.index_to_word = index_to_word

    def encode(self, word):
        if word not in self.word_to_index:
            word = self.unknown
        return self.word_to_index[word]

    def decode(self, index):
        return self.index_to_word[index]

