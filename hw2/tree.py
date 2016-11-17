import random
UNK = 'UNK'

class Node:  # a node in the tree
    #def __init__(self, label, word=None):
    def __init__(self, word=None):
        self.word = word
        self.parent = None
        self.left = None
        self.right = None
        self.isLeaf = False


class Tree:
    def __init__(self, treeString, label, openChar='(', closeChar=')'):
        tokens = []
        self.labels = label
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        self.num_words = len(self.get_words())

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words

    def parse(self, tokens, parent=None):
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"
        split = 2
        countOpen = countClose = 0

        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1
        node = Node()
        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower()  # lower case?
            node.isLeaf = True
            return node
        node.left = self.parse(tokens[2:split], parent=node)
        node.right = self.parse(tokens[split:-1], parent=node.left)
        return node


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


def loadTrees(label,dataSet='train'):
    file = 'trees/%s' % dataSet
    print ("Loading %s .." % dataSet)
    with open(file, 'rt') as fid:
        trees = [Tree(l,label) for l in fid.readlines()]
    return trees

def simplified_data():
    ptrees=loadTrees(1,'positiveBinaryTree')
    ntrees=loadTrees(0,'negativeBinaryTree')
    ttrees=loadTrees(0,'testingBinaryTree')

    test_trees=ttrees[:]
    train=list()
    for i in range(len(ptrees)):
        train.append(ptrees[i])
        train.append(ntrees[i])

    random.shuffle(train)
    dsize = len(train)//20
    dev=train[:-dsize]
    train=train[-dsize:]
    return train,ttrees,dev

def load_test_data():
    file = './testingBinaryTree'
    with open(file, 'rt') as fid:
        trees = [Tree(l,0) for l in fid.readlines()]
    return trees
