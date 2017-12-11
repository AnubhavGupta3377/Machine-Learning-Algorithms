import math

from collections import defaultdict

def log2(num):
    if num == 0:
        return 0
    return math.log(num, 2)
    
class Entropy:
    
    '''
    1. labels : list of all the labels of all the examples
    2. features : list of list of feature values of all the examples
    3. values : dictionary that maps every feature to the set of 
       possible values it can take
    4. n : #features + 1(label)
    '''
    def __init__(self, labels=[], features=[], values=[], n=0):
        self.labels = labels[:]
        self.features = features[:]
        self.values = values
        self.n = n
        
    '''
    Calculate the entropy given number of positive and negative
    examples.
    i.e., Compute H(X)
    '''
    def getEntropy(self, numPos, numNeg):
        total = numPos + numNeg
        if numPos == 0 or numNeg == 0:
            return 0
        pPos = float(numPos) / total
        pNeg = float(numNeg) / total
        entropy = -(pPos * log2(pPos) + pNeg * log2(pNeg))
        return entropy
        
    '''
    Calculate the specific conditional entropy at node 'node' given 
    that decision is made on the attribute 'attr = value'.
    i.e., Compute H(X | attr = value)
    '''
    def getSpecificCondEntropy(self, node, attr, value):
        numPos = 0
        numNeg = 0
        for example in node.examples:
            if self.features[example][attr] == value:
                if self.labels[example] == 0:
                    numNeg += 1
                else:
                    numPos += 1
        return self.getEntropy(numPos, numNeg)
        
    '''
    Calculate the conditional entropy at node 'node' given that 
    decision is made on the attribute 'attr'.
    i.e., Compute H(X | attr)
    '''
    def getConditionalEntropy(self, node, attr):
        entropy = 0
        counts = defaultdict(int)
        for example in node.examples:
            counts[self.features[example][attr]] += 1
        totalExamples = len(node.examples)
        if totalExamples == 0:
            return 0
        for value in self.values[attr]:
            entropy += ((float(counts[value])) / totalExamples) * self.getSpecificCondEntropy(node, attr, value)
        return entropy
        
    '''
    Calculate the mutual information (information gain) at node 'node'
    given that decision is made on the attribute 'attr'.
    i.e., Compute I(X, attr) = H(X) - H(X | attr)
    '''
    def getMutualInformation(self, node, attr):
        numPos = 0
        numNeg = 0
        for example in node.examples:
            if self.labels[example] == 0:
                numNeg += 1
            else:
                numPos += 1
        return self.getEntropy(numPos, numNeg) - self.getConditionalEntropy(node, attr)
        
    '''
    Find the best attribute to split at the current node.
    '''
    def getBestAttr(self, node):
        bestGain = 0
        bestAttr = 0
        for i in range(self.n-1):
            gain = self.getMutualInformation(node, i)
            if gain > bestGain:
                bestAttr = i
                bestGain = gain
        return (bestAttr, bestGain)