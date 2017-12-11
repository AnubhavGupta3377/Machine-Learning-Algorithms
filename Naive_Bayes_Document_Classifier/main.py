import numpy as np
import matplotlib.pyplot as plt
import csv
from naiveBayes import NaiveBayes

'''
  Get classification accuracy of the model
'''
def getAccuracy(y_original, y_predicted):
    match = (y_original == y_predicted)
    return float(match.sum()) / len(match)

'''
  Print the confusion matrix for the model
'''
def printConfusionMatrix(y_original, y_predicted, labels):
    matrix = [[0 for i in range(len(labels))] for j in range(len(labels))]
    numTest = len(y_predicted)
    for i in range(numTest):
        matrix[int(y_original[i]-1)][int(y_predicted[i]-1)] += 1
    row_format ="{:>4}" * (len(labels) + 1)
    print row_format.format("", *labels)
    for label, row in zip(labels, matrix):
        print row_format.format(label, *row)
        
if __name__ == '__main__':
    
    labels = ['']
    vocabulary = ['']
    X_train = []
    X_test = []
    
    print 'Reading data from files...'
    
    vocabFile = open('Data/vocabulary.txt')
    for line in vocabFile:
        vocabulary.append(line.strip())
    V = len(vocabulary) - 1
    vocabFile.close()
        
    labelsFile = open('Data/newsgrouplabels.txt')
    for line in labelsFile:
        labels.append(line.strip())
    numY = len(labels) - 1
    labelsFile.close()
    
    trainLabelsFile = open('Data/train.label', 'r')
    y_train = np.loadtxt(trainLabelsFile)
    y_train = np.array([0]+list(y_train))
    numTrain = y_train.shape[0] - 1
    trainLabelsFile.close()
    
    testLabelsFile = open('Data/test.label', 'r')
    y_test = np.loadtxt(testLabelsFile)
    numTest = y_test.shape[0] - 1
    testLabelsFile.close()
    
    trainDataFile = open('Data/train.data', 'r')
    trainReader = csv.reader(trainDataFile, delimiter = ' ')
    for row in trainReader:
        docId, wordId, count = map(int, row)
        X_train.append([docId, wordId, count])
    X_train = np.array(X_train)
    trainDataFile.close()
        
    testDataFile = open('Data/test.data', 'r')
    testReader = csv.reader(testDataFile, delimiter = ' ')
    for row in testReader:
        docId, wordId, count = map(int, row)
        X_test.append([docId, wordId, count])
    X_test = np.array(X_test)
    testDataFile.close()
    
    alpha = 1.0 / V
    print 'Initializing Naive Bayes classifier...'
    NBClassifier = NaiveBayes(V, numY)
    print 'Training the model...'
    NBClassifier.train(X_train, y_train, alpha)
    print 'Making predictions on test data...'
    y_predicted = NBClassifier.predict(X_test)
        
    # Solution for question 3.2
    print 'Accuracy of Naive Bayes classifier: ' + str(getAccuracy(y_test, y_predicted))
    print
    #print 'Confusion Matrix:'
    #print
    #printConfusionMatrix(y_test, y_predicted, range(1,21))
    #print
    
    
    # Solution for question 3.4
    '''    
    print 'Experimenting with different values of alpha...'
    p = 0.00001
    alphas = [p, 2*p, 3*p, 4*p, 5*p, 6*p, 7*p, 8*p, 9*p, 10*p, 20*p,
              30*p, 40*p, 50*p, 60*p, 70*p, 80*p, 90*p, 100*p, 200*p,
              300*p, 400*p, 500*p, 600*p, 700*p, 800*p, 900*p, 1000*p,
              2000*p, 3000*p, 4000*p, 5000*p, 6000*p, 7000*p, 8000*p,
              9000*p, 10000*p, 20000*p, 30000*p, 40000*p, 50000*p,
              60000*p, 70000*p, 80000*p, 90000*p, 100000*p]
    accuracies = []
    for alpha in alphas:
        print 'Setting alpha = ' + str(alpha)
        NBClassifier.train(X_train, y_train, alpha)
        y_predicted = NBClassifier.predict(X_test)
        accuracy = getAccuracy(y_test, y_predicted)
        accuracies.append(accuracy)
        print 'Accuracy = ' + str(accuracy)
        
    plt.semilogx(alphas, accuracies, 'b-')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.show()
    '''