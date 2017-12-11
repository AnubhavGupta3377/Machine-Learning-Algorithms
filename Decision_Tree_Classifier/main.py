import ReadData
import DTree
from entropy import *

EPSILON = 0.005
    
if __name__ == '__main__':
    
    # Reading training data and storing values in variables
    trainObj = ReadData.ReadData()
    trainObj.readData('Data/noisy10_train.ssv')
    
    trainLabels = trainObj.labels[:]
    numTrainExamples = trainObj.numExamples
    trainFeatures = trainObj.features
    trainValues = trainObj.values
    
    n = trainObj.n
    names = trainObj.names
    nums = trainObj.nums
    outputLabel = trainObj.outputLabel
    
    # Reading testing data
    testObj = ReadData.ReadData()
    testObj.readData('Data/noisy10_test.ssv')
    
    testLabels = testObj.labels[:]
    numTestExamples = testObj.numExamples
    testFeatures = testObj.features
    testValues = testObj.values

    # Reading validation data
    validObj = ReadData.ReadData()
    validObj.readData('Data/noisy10_valid.ssv')
    
    validLabels = validObj.labels[:]
    numValidExamples = validObj.numExamples
    validFeatures = validObj.features
    validValues = validObj.values
    
    # Building root node of decision tree
    # This node contains all the training examples
    node = DTree.DTNode(numTrainExamples, sum(trainLabels), len(trainLabels) - sum(trainLabels), [], range(numTrainExamples))
    decisionTree = DTree.DecisionTree(node, 0, 0)
    
    print
    print '----------------------------------'
    print '     Building Decision Tree'
    print '----------------------------------'
    print
    
    decisionTree.trainDTree(trainLabels, trainFeatures, trainValues, n, names)
    depth = decisionTree.depth
    numNodes = decisionTree.numNodes
    
    print
    print '-------------------------------------------------'
    print '       Computing Decision Tree Statistics'
    print '-------------------------------------------------'
    print
    
    print 'Depth of decision tree: ' + str(depth)
    print
    print 'Number of nodes in the decision tree: ' + str(numNodes)
    print

    numNodes = [0 for i in range(depth)]
    trainAccuracy = [0 for i in range(depth)]
    for i in range(1, depth+1):
        numNodes[i-1] = decisionTree.numNodesAtDepth(i)
        trainAccuracy[i-1] = decisionTree.findAccuracy(trainFeatures, trainLabels, i)
        
    testAccuracy = [0 for i in range(depth)]
    for i in range(1, depth+1):
        testAccuracy[i-1] = decisionTree.findAccuracy(testFeatures, testLabels, i)
        
    print '--------------------------------------------'
    print '%-10s %-10s %-10s %-10s' %('Max', '# of', 'Correct', 'Correct')
    print '%-10s %-10s %-10s %-10s' %('depth', 'nodes', 'train %', 'test %')
    print '--------------------------------------------'
    for i in range(1, depth+1):
        print '%-10d %-10d %-10.2f %-10.2f' %(i, numNodes[i-1], round(trainAccuracy[i-1],2), round(testAccuracy[i-1],2))
    print '--------------------------------------------'
    print '%-10s %-10d %-10.2f %-10.2f' %('FINAL', numNodes[depth-1], round(trainAccuracy[depth-1],2), round(testAccuracy[depth-1],2))
    print '--------------------------------------------'
    
    print
    print '-------------------------------------------------'
    print '            Pruning the Decision Tree'
    print '-------------------------------------------------'
    print
    
    decisionTree.pruneTree(node, validFeatures, validLabels, EPSILON, 1)
    prunedDepth = decisionTree.depth
    prunedNumNodes = decisionTree.numNodes
    
    print
    print str.format('Pruning reduced decision tree size from {0} nodes to {1} nodes', numNodes[depth-1], prunedNumNodes)
    print
    
    print
    print '--------------------------------------------------------'
    print '    Computing Decision Tree Statistics after pruning'
    print '--------------------------------------------------------'
    print
    
    print 'Depth of decision tree: ' + str(prunedDepth)
    print
    print 'Number of nodes in the decision tree: ' + str(prunedNumNodes)
    print

    numNodes = [0 for i in range(prunedDepth)]
    trainAccuracy = [0 for i in range(prunedDepth)]
    for i in range(1, prunedDepth+1):
        numNodes[i-1] = decisionTree.numNodesAtDepth(i)
        trainAccuracy[i-1] = decisionTree.findAccuracy(trainFeatures, trainLabels, i)
        
    testAccuracy = [0 for i in range(prunedDepth)]
    for i in range(1, prunedDepth+1):
        testAccuracy[i-1] = decisionTree.findAccuracy(testFeatures, testLabels, i)
    
    print '--------------------------------------------'
    print '%-10s %-10s %-10s %-10s' %('Max', '# of', 'Correct', 'Correct')
    print '%-10s %-10s %-10s %-10s' %('depth', 'nodes', 'train %', 'test %')
    print '--------------------------------------------'
    for i in range(1, prunedDepth+1):
        print '%-10d %-10d %-10.2f %-10.2f' %(i, numNodes[i-1], round(trainAccuracy[i-1],2), round(testAccuracy[i-1],2))
    print '--------------------------------------------'
    print '%-10s %-10d %-10.2f %-10.2f' %('FINAL', numNodes[prunedDepth-1], round(trainAccuracy[prunedDepth-1],2), round(testAccuracy[prunedDepth-1],2))
    print '--------------------------------------------'