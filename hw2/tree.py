import random
UNK = 'UNK'
# This file contains the dataset in a useful way. We populate a list of
# Trees to train/test our Neural Nets such that each Tree contains any
# number of Node objects.

# The best way to get a feel for how these objects are used in the program is to drop pdb.set_trace() in a few places throughout the codebase
# to see how the trees are used.. look where loadtrees() is called etc..


class Node:  # a node in the tree
    def __init__(self, word=None):
        # self.label = label
        self.word = word
        self.parent = None  # reference to parent
        self.left = None  # reference to left child
        self.right = None  # reference to right child
        # true if I am a leaf (could have probably derived this from if I have
        # a word)
        self.isLeaf = False
        # true if we have finished performing fowardprop on this node (note,
        # there are many ways to implement the recursion.. some might not
        # require this flag)


class Tree:

    def __init__(self, treeString, label, openChar='(', closeChar=')'):
        tokens = []
        self.open = '('
        self.close = ')'
        for toks in treeString.strip().split():
            tokens += list(toks)
        self.root = self.parse(tokens)
        # get list of labels as obtained through a post-order traversal
        self.label = label
        # self.labels = get_labels(self.root)
        # self.num_words = len(self.labels)

    def parse(self, tokens, parent=None):
        if len(tokens) == 0:
            return None
        assert tokens[0] == self.open, "Malformed tree"
        assert tokens[-1] == self.close, "Malformed tree"

        split = 2  # position after open and label
        countOpen = countClose = 0


        if tokens[split] == self.open:
            countOpen += 1
            split += 1
        # Find where left child and right child split
        while countOpen != countClose:
            if tokens[split] == self.open:
                countOpen += 1
            if tokens[split] == self.close:
                countClose += 1
            split += 1

        # New node
        node = Node()  # zero index labels

        node.parent = parent

        # leaf Node
        if countOpen == 0:
            node.word = ''.join(tokens[2:-1]).lower()  # lower case?
            node.isLeaf = True
            return node

        node.left = self.parse(tokens[2:split], parent=node)
        right = self.parse(tokens[split:-1], parent=node)
        if right is not None:
            node.right = right
            return node
        return node.left
        # node.right = self.parse(tokens[split:-1], parent=node)

    def get_words(self):
        leaves = getLeaves(self.root)
        words = [node.word for node in leaves]
        return words


def leftTraverse(node, nodeFn=None, args=None):
    """
    Recursive function traverses tree
    from left to right. 
    Calls nodeFn at each node
    """
    if node is None:
        return
    leftTraverse(node.left, nodeFn, args)
    leftTraverse(node.right, nodeFn, args)
    nodeFn(node, args)


def getLeaves(node):
    if node is None:
        return []
    if node.isLeaf:
        return [node]
    else:
        return getLeaves(node.left) + getLeaves(node.right)


# def get_labels(node):
#     if node is None:
#         return []
#     return get_labels(node.left) + get_labels(node.right) + [node.label]


def clearFprop(node, words):
    node.fprop = False


#modified done
def loadTrees(label, dataSet='train'):
    # load pure text from tree file
    """
    Loads training trees. Maps leaf node words to word ids.
    """
    # file = 'trees/%s.txt' % dataSet
    file=dataSet
    # print "Loading %s trees.." % dataSet
    with open(file, 'r') as fid:
        # sentence = ''
        # trees = []
        # f = fid.read().splitlines()
        # for line in f:
        #     sentence += line
        #     if '(. .)' in line:
        #         trees.append(Tree(sentence,label=1))
        #         sentence = ''
        trees = [Tree(l,label=label) for l in fid.readlines() if len(l) > 0]

    # tree string in trees (which is a list)
    return trees


def simplified_data(pos_f, neg_f, test_f):
    rndstate = random.getstate()
    random.seed(0)
    pos_trees = loadTrees(dataSet=pos_f, label =1)
    neg_trees = loadTrees(dataSet=neg_f, label =0)
    pos_trees = sorted(pos_trees, key=lambda t: len(t.get_words()))
    pos_trees = sorted(neg_trees, key=lambda t: len(t.get_words()))
    train =  pos_trees[:] + neg_trees[:]
    test_trees = loadTrees(dataSet=test_f, label =1)
    test = test_trees[:]

    random.shuffle(train)
    random.shuffle(test)
    random.setstate(rndstate)
    return train, test
