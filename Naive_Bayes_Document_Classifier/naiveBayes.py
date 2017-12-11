import numpy as np
from math import log

class NaiveBayes:
    def __init__(self, V, numY):
        self.V = V
        self.numY = numY
        self.probY = []
        self.classCondProb = []
        
    def train(self, X, y, alpha = 0.001):
        numTrain = len(y) - 1
        self.probY = [1 for i in range(self.numY+1)]
        for i in range(1,self.numY+1):
            self.probY[i] = y.tolist().count(i) / float(numTrain)
        numCoOccurence = np.array([[0 for w in range(self.V+1)] for label in range(self.numY+1)], dtype = float)
        for row in X:
            docId, wordId, count = row
            numCoOccurence[int(y[docId])][wordId] += count
        self.classCondProb = np.array([[1 for w in range(self.V+1)] for label in range(self.numY+1)], dtype = float)
        for label in range(1,self.numY+1):
            denominator = numCoOccurence[label].sum()
            self.classCondProb[label] = (numCoOccurence[label] + alpha*np.ones(self.V+1)) / (denominator + alpha*self.V)
            
    def predict(self, X):
        numTest = X[:,0].max()
        y_predicted = np.zeros((numTest+1))
        probYforD = np.array([[log(self.probY[y]) for y in range(self.numY+1)] for d in range(numTest+1)], dtype = float)
        for row in X:
            docId, wordId, count = row
            probYforD[docId] += count * np.log(np.transpose(self.classCondProb[:,wordId]))
        for docId in range(1,numTest+1):
            y = probYforD[docId][1:].argmax() + 1
            y_predicted[docId] = y
        return y_predicted[1:]