from nltk.tree import Tree
import nltk.treetransforms
from copy import deepcopy
import sys

pos = open('data/training_data.pos.tree','r').read().split('\n')
neg = open('data/training_data.neg.tree','r').read().split('\n')

pos_out = open('data/raw_tree.pos','w')
neg_out = open('data/raw_tree.neg','w')


sentence = ''
sys.stdout = pos_out
for line in pos:
    sentence += line
    if '(. .)' in line:
        tree = Tree.fromstring(sentence)
        nltk.treetransforms.chomsky_normal_form(tree)
        print(tree)
        sentence = ''
pos_out.close()

sentence = ''
sys.stdout = neg_out
for line in neg:
    sentence += line
    if '(. .)' in line:
        tree = Tree.fromstring(sentence)
        nltk.treetransforms.chomsky_normal_form(tree)
        print(tree)
        sentence = ''
neg_out.close()

def change_label(line):
    newline = ''
    need_copy = False
    for s in line:
        if need_copy:
            newline += s
        if s == ' ':
            need_copy = True
        elif s == '(':
            newline += '0 '
            need_copy = False
    return newline


pos = open('data/raw_tree.pos','r').read().splitlines()
neg = open('data/raw_tree.neg','r').read().splitlines()

pos_out = open('data/bin_tree.pos','w')
neg_out = open('data/bin_tree.neg','w')

sentence = ''
for line in pos:
    sentence += ' '+line.lstrip()
    if '(. .)' in line:
        pos_out.write(change_label(sentence)+'\n')
        sentence = ''
pos_out.close()

sentence = ''
for line in neg:
    sentence += ' '+line.lstrip()
    if '(. .)' in line:
        neg_out.write(change_label(sentence)+'\n')
        sentence = ''
pos_out.close()

