# Naive Bayes Classifier
- This is the code of Naive Bayes classifier for classifying documents in one of 20 newsgroups.

## Implementation Details
- Say we have a document **D** containing **n** words; call the words ![](https://latex.codecogs.com/gif.latex?X_1%2C%5Ccdots%2CX_n).
- The value of random variable ![](https://latex.codecogs.com/gif.latex?X_i) is the word found in position i in the document.
- Here we wish to predict the label **Y** of the document, which can be one of **m** categories.
- We used the model:
	
	![](https://latex.codecogs.com/gif.latex?P%28Y%20%7C%20X_1%2C%5Ccdots%2CX_n%29%20%5Cpropto%20P%28X_1%20...X_n%20%7C%20Y%20%29%7EP%28Y%29%20%3D%20P%28Y%29%20%5Cprod_%7Bi%3D1%7D%5Em%20P%28X_i%20%7C%20Y%29)
	
	with additional assumption that
	
	![](https://latex.codecogs.com/gif.latex?%5Cforall%20i%2Cj%7E%7E%20P%20%28X_i%20%7C%20Y%29%20%3D%20p%28X_j%20%7C%20Y%29)
									 
- In our implementation, we estimated ![](https://latex.codecogs.com/gif.latex?P%28Y%29) using the MLE, and estimated ![](https://latex.codecogs.com/gif.latex?P%28X%7CY%29) using a MAP estimate with the prior distribution ![](https://latex.codecogs.com/gif.latex?%5Ctext%7BDirichlet%7D%281&plus;%5Calpha%2C%5Ccdots%2C%201%20&plus;%20%5Calpha%29),

	where  ![](https://latex.codecogs.com/gif.latex?%5Calpha%20%3D%20%5Cfrac%7B1%7D%7B%7CV%7C%7D) and V is vocabulary.
    
- We also experimented with different values of ![](https://latex.codecogs.com/gif.latex?%5Calpha).

## Author
- **Anubhav Gupta**

## Prerequisites
- Python 2.7
