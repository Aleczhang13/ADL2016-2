# coding=UTF-8
from nltk.tree import Tree
import nltk.treetransforms
from copy import deepcopy
import sys
import io


def write_raw(in_f, out_f):
    sentence = ''
    sys.stdout = out_f
    for line in in_f:
        sentence += line
        if '(. .)' in line:
            tree = Tree.fromstring(sentence)
            nltk.treetransforms.chomsky_normal_form(tree)
            print(tree)
            sentence = ''
    out_f.close()

pos = open('data/training_data.pos.tree','r').read().split('\n')
neg = open('data/training_data.neg.tree','r').read().split('\n')
test = io.open('data/testing_data.txt.tree','r',encoding='utf8').readlines()

pos_out = open('data/raw_tree.pos','w')
neg_out = open('data/raw_tree.neg','w')
test_out = open('data/raw_tree.test','w')
write_raw(pos, pos_out)
write_raw(neg, neg_out)
write_raw(test, test_out)


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

def clean_tree(in_f, out_f):
    sentence = ''
    for line in in_f:
        sentence += ' '+line.lstrip()
        if '(. .)' in line:
            out_f.write(change_label(sentence)+'\n')
            sentence = ''
    out_f.close()


pos = open('data/raw_tree.pos','r').read().splitlines()
neg = open('data/raw_tree.neg','r').read().splitlines()
test = open('data/raw_tree.test','r').read().splitlines()
pos_out = open('data/bin_tree.pos','w')
neg_out = open('data/bin_tree.neg','w')
test_out = open('data/bin_tree.test','w')


clean_tree(pos, pos_out)
clean_tree(neg, neg_out)
clean_tree(test, test_out)
