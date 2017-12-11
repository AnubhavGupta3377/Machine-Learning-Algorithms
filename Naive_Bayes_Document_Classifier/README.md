# Naive Bayes Classifier
- This is the code of Naive Bayes classifier for classifying documents in one of 20 newsgroups.

## Implementation Details
- Say we have a document **D** containing **n** words; call the words {X_1, ..., X_n}.
    The value of random variable X_i is the word found in position i in the document.
    Here we wish to predict the label Y of the document, which can be one of m categories.
  
  - We used the model:
		
		P (Y | X_1 ...X_n ) ∝ P(X_1 ...X_n | Y ) P(Y)
							= P(Y) product(P(X_i | Y))
									 i
									 
	with additional assumption that
		
		∀i,j	P (X_i | Y) = p(X_j | Y)
									 
  - In our implementation, we estimated P (Y) using the MLE, and estimated P (X|Y)
    using a MAP estimate with the prior distribution Dirichlet(1 + ALPHA, ..., 1 + ALPHA),
    where ALPHA = 1/|V| and V is vocabulary.
    
  - We also experimented with different values of ALPHA.

## Author
- **Anubhav Gupta**
