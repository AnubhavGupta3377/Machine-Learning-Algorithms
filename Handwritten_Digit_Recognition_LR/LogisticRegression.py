import numpy as np
import math
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self,w=np.array([])):
        self.w = w
        
    def prob_yk_given_x(self, X, K, k):
        n, d = X.shape
        numer = np.exp(np.dot(X,self.w[k]))
        denom = np.ones(n)
        denom = denom + np.exp(np.dot(X, self.w.transpose())).sum(axis=1)
        return numer / denom
    
    def predict(self, X):
        K = self.w.shape[0] - 1
        n, d = X.shape
        maxProb = np.zeros(n)
        y = np.zeros(n, dtype=int)
        for k in range(1,K+1):
            prob = self.prob_yk_given_x(X,K,k)
            maxProb = np.maximum(prob, maxProb)
            new_y = np.ones(n, dtype=int) * k
            eq = (maxProb == prob)
            ne = (maxProb != prob)
            y = eq*new_y + ne*y
        return y
        
    def getAccuracy(self, X, y):
        predicted_y = self.predict(X)
        return (predicted_y == y).sum() / float(len(y))
    
    def getCondLikelihood(self, X, y, K):
        n, d = X.shape
        y = y.astype(int)
        inner_sum = np.ones(n) + np.exp(np.dot(X, self.w.transpose())).sum(axis=1)
        inner_sum = np.log(inner_sum)
        newW = self.w[y]
        result = (X * newW).sum(axis=1) - inner_sum
        likelihood = result.sum()
        return likelihood
    
    # Commented code in this function is for solution of 5(a)
    def train_model(self, X, y, X_te, y_te, lamda=0, alpha = 0.0001):
        labels = np.unique(y)
        K = len(labels)
        n = X.shape[0]
        d = X.shape[1]
        self.w = np.zeros((K+1, d))
        
        I = [0 for k in range(K+1)]
        for k in range(1,K+1):
            temp = np.ones((n,1)) * k
            I[k] = (temp == y).nonzero()
            
        numSteps = 0
        
        #x_axis = [i for i in range(2001)]
        #y_train = [self.getAccuracy(X,y)]
        y_test = [self.getAccuracy(X_te, y_te)]
        #likelihood = [self.getCondLikelihood(X,y,K)]
        while numSteps < 2000:
            numSteps += 1
            if numSteps % 100 == 0:
                print 'Number of iterations = ' + str(numSteps)
            for k in range(1,K):
                eq = (y == (np.ones(n) * k))
                gradientL = np.dot(X.transpose(), eq - self.prob_yk_given_x(X,K,k)) - lamda*self.w[k]
                self.w[k] = self.w[k] + alpha * gradientL
                
            #y_train.append(self.getAccuracy(X,y))
            y_test.append(self.getAccuracy(X_te, y_te))
            #likelihood.append(self.getCondLikelihood(X,y,K))
        
        '''
        plt.figure(1, figsize=(10,7.5))
        plt.plot(x_axis,y_train)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Training Accuracy')
        plt.savefig('Iterations_vs_train_acc_lambda_1.png')
        plt.show()
        
        plt.figure(2, figsize=(10,7.5))
        plt.plot(x_axis,y_test)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Testing Accuracy')
        plt.savefig('Iterations_vs_test_acc_lambda_1.png')
        plt.show()
        
        plt.figure(3, figsize=(10,7.5))
        plt.plot(x_axis,likelihood)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Objective Value')
        plt.savefig('Iterations_vs_objective_value_lambda_1.png')
        plt.show()
        '''
        
        print 'Final Test Accuracy: ' + str(y_test[-1])