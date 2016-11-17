from nltk.tree import Tree
import nltk as nl
from nltk.draw.tree import draw_trees
from nltk import tree, treetransforms
from copy import deepcopy
import sys


def chomsky_normal_form(tree, factor="right", horzMarkov=None, vertMarkov=0, childChar="|", parentChar="^"):
    if horzMarkov is None: horzMarkov = 999
    nodeList = [(tree, ['0'])]
    while nodeList != []:
        node, parent = nodeList.pop()
        if isinstance(node,Tree):
            node.set_label("0")
            if vertMarkov != 0 and node != tree and isinstance(node[0],Tree):
                node.set_label("0")
            if len(node)==1:
                if (len(node.leaves()))==1:
                    word = node.leaves()[0]
                    node[0] = word
                elif len(node[0]) != 1:
                    node.append(node[0][1])
                    node[0] = node[0][0]
                elif len(node[0][0])!=1:
                    node.append(node[0][0][1])
                    node[0]=node[0][0][0]
            for child in node:
                nodeList.append((child, "0"))
            if len(node) > 2:
                childNodes = ['0' for child in node]
                nodeCopy = node.copy()
                for i in range(1,len(nodeCopy) - 1):
                    if factor == "right":
                        newNode = Tree("0", [])
                        node[0:] = [nodeCopy.pop(0), newNode]
                    else:
                        newNode = Tree("0", [])
                        node[0:] = [newNode, nodeCopy.pop()]
                    node = newNode
                node[0:] = [child for child in nodeCopy]

with open(sys.argv[1],'rt') as fin:
    raw_data = fin.read().split('\n\n')
    del raw_data[len(raw_data)-1]
    TT = [Tree.fromstring(t) for t in raw_data]

with open('testingBinaryTree','wt') as fout:
    for i,x in enumerate(TT):
        chomsky_normal_form(x)
        a=x.pformat().split()
        b=' '.join(a)
        fout.write(b+'\n')

