# 10-701 Machine Learning, Spring 2011: Homework 3
# Solution code for question 5 of Section 2

import numpy as np
import LogisticRegression

X_tr = np.loadtxt(open("Data/tr_X.txt","rb"),delimiter=",")
y_tr = np.loadtxt(open("Data/tr_y.txt","rb"),delimiter=",")
X_te = np.loadtxt(open("Data/te_X.txt","rb"),delimiter=",")
y_te = np.loadtxt(open("Data/te_y.txt","rb"),delimiter=",")

'''
# Let's see the digit represented by this vector

plt.imshow(X_tr[1000].reshape((16,16)))
plt.gray()
plt.show()
'''

model = LogisticRegression.LogisticRegression()

# Pass different values of lambda for solution of 5(b)
model.train_model(X_tr, y_tr, X_te, y_te, lamda=1000)