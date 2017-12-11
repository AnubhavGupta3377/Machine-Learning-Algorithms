from collections import defaultdict, deque
import entropy

class DecisionTree:
    def __init__(self, root, depth, numNodes, trainFeatures = [], names = defaultdict(str)):
        self.root = root
        self.depth = depth
        self.numNodes = numNodes
        self.trainFeatures = trainFeatures[:]
        self.names = names
        
    '''
    Build the decision tree using training data.
    names maps attribute number to its name.
    '''
    def trainDTree(self, trainLabels, trainFeatures, trainValues, n, names):
        self.trainFeatures = trainFeatures
        self.names = names
        root = self.root
        root.splitNode(trainLabels, trainFeatures, trainValues, n, names)
        self.depth, self.numNodes = root.getSizeofTree()
    
    '''
    Predict output label at the current node.
    '''
    def predictAtNode(self, node, example, d, maxDepth = -1):
        if maxDepth == -1:
            maxDepth = self.depth
        if not node.isInternalNode():
            return node.getLabel()
        elif d == maxDepth:
            return node.getLabel()
        for child in node.children:
            if len(child.examples) > 0:
                if self.trainFeatures[child.examples[0]][node.attr] == example[node.attr]:
                    return self.predictAtNode(child, example, d+1, maxDepth)
    
    '''
    Predict output label for this example.
    '''
    def predict(self, example, maxDepth = -1):
        return self.predictAtNode(self.root, example, 1, maxDepth)
        
    '''
    Find accuracy of the decision tree while considering the nodes 
    till depth maxDepth.
    '''
    def findAccuracy(self, features, labels, maxDepth = -1):
        if maxDepth == -1:
            maxDepth = self.depth
        predictions = []
        for example in features:
            y = self.predict(example, maxDepth)
            predictions.append(y)
        matches = 0
        for j in range(len(labels)):
            if predictions[j] == labels[j]:
                matches += 1
        accuracy = 100 * (float(matches) / len(labels))
        return accuracy
        
    '''
    Count the number of nodes at depth atmost maxDepth in the
    decision tree.
    '''
    def numNodesAtDepth(self, maxDepth):
        root = self.root
        q = deque()
        numNodes = 0
        q.append((root, 1))
        while len(q) > 0:
            node, d = q.popleft()
            if d > maxDepth:
                return numNodes
            numNodes += 1
            for childNode in node.children:
                q.append((childNode, d+1))
        return numNodes
        
    '''
    This method performs pruning of decision tree. It first prunes current 
    node, then prunes its children (after you decided to keep this node).
    '''
    def pruneTree1(self, node, features, labels, EPSILON):
        if len(node.children) == 0:
            return
        depth = self.depth
        accBeforePrun = self.findAccuracy(features, labels, depth)
        children = node.children
        attr = node.attr
        node.children = []
        node.attr = -1
        accAfterPrun = self.findAccuracy(features, labels, depth)
        if (accAfterPrun - accBeforePrun) >= (EPSILON * 100):
            print str.format('Pruning     : {0} (accuracy: {1} -> {2})', self.names[attr], round(accBeforePrun,6), round(accAfterPrun,6))
            return
        print str.format('Not pruning : {0} (accuracy: {1} -> {2})', self.names[attr], round(accBeforePrun,6), round(accAfterPrun,6))
        node.children = children
        node.attr = attr
        for childNode in node.children:
            self.pruneTree1(childNode, features, labels, EPSILON)
    
    '''
    This method performs pruning of decision tree. It first recursively checks 
    the child nodes of current node for pruning, then checks current node.
    '''
    def pruneTree2(self, node, features, labels, EPSILON):
        if len(node.children) == 0:
            return
        rootNode = self.root
        for childNode in node.children:
            self.pruneTree2(childNode, features, labels, EPSILON)
        depth, numNodes = rootNode.getSizeofTree()
        accBeforePrun = self.findAccuracy(features, labels, depth)
        children = node.children
        attr = node.attr
        node.children = []
        node.attr = -1
        accAfterPrun = self.findAccuracy(features, labels, depth)
        if (accAfterPrun - accBeforePrun) >= (EPSILON * 100):
            print str.format('Pruning     : {0} (accuracy: {1} -> {2})', self.names[attr], round(accBeforePrun,6), round(accAfterPrun,6))
            return
        print str.format('Not pruning : {0} (accuracy: {1} -> {2})', self.names[attr], round(accBeforePrun,6), round(accAfterPrun,6))
        node.children = children
        node.attr = attr
        return
        
    '''
    Prune the decision tree.
    If method == 1, then perform pruning in top-down manner, Otherwise 
    perform pruning in bottom-up manner.
    '''
    def pruneTree(self, node, features, labels, EPSILON, method = 1):
        if method == 1:
            self.pruneTree1(node, features, labels, EPSILON)
        else:
            self.pruneTree2(node, features, labels, EPSILON)
        self.depth, self.numNodes = self.root.getSizeofTree()
        
class DTNode:

    '''
    Defines the structure of the decision tree node.
    1. numExamples : Number of examples at this node
    2. numPos : Number of positive examples at this node
    3. numNeg : Number of negative examples at this node
    4. children : Children nodes of this node
    5. examples : examples at this node
    '''
    def __init__(self, numExamples=0, numPos=0, numNeg=0, children=[], examples=[]):
        self.numExamples = numExamples
        self.numPos = numPos
        self.numNeg = numNeg
        self.children = children[:]
        self.examples = examples[:]
        self.attr = -1
        
    '''
    Perform splitting at current node based on the the attrubite 'attr'.
    '''
    def splitNode(self, labels, features, values, n, names):
        entropyObj = entropy.Entropy(labels, features, values, n)
        attr, gain = entropyObj.getBestAttr(self)
        self.attr = attr
        
        if gain <= 0:
            #print 'Gain is Not sufficient'
            return
        print str.format('Selected attribute "{0}" (Gain = {1})', names[attr], round(gain,6))
        
        self.children = [DTNode() for i in range(len(values[attr]))]
        exampleToChild = defaultdict(int)
        childNum = 0
        
        for example in self.examples:
            value = features[example][attr]
            if exampleToChild[value] == 0:
                childNum += 1
                exampleToChild[value] = childNum
            child = exampleToChild[value] - 1
            node = self.children[child]
            node.examples.append(example)
            node.children = []

        for childNode in self.children:
            numPos = 0
            childNode.numExamples = len(childNode.examples)
            for example in childNode.examples:
                if labels[example] == 1:
                    numPos += 1
            childNode.numPos = numPos
            childNode.numNeg = childNode.numExamples - childNode.numPos
        
        for childNode in self.children:
            childNode.splitNode(labels, features, values, n, names)
        
    '''
    Return the depth of decision tree and number of nodes in
    the tree.
    '''
    def getSizeofTree(self):
        q = deque()
        numNodes = 0
        depth = 1
        q.append((self, 1))
        while len(q) > 0:
            node, d = q.popleft()
            numNodes += 1
            if d > depth:
                depth = d
            for childNode in node.children:
                q.append((childNode, d+1))
        return (depth, numNodes)
       
    '''
    Check if a node is internal node in the decision tree.
    '''
    def isInternalNode(self):
        if len(self.children) != 0:
            return True
        return False
       
    '''
    Get the label associated with current node.
    '''
    def getLabel(self):
        if self.numNeg > self.numPos:
            return 0
        if self.numNeg < self.numPos:
            return 1
        return 1