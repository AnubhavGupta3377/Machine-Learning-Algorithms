from collections import defaultdict

class ReadData:
    
    '''
    1. labels : list of all the labels of all the examples
    2. features : list of list of feature values of all the examples
    3. values : dictionary that maps every feature to the set of 
       possible values it can take
    4. numExamples : number of (training) examples in the dataset
    5. n : #features + 1(label)
    6. names : maps every number to associated name of the attribute
    7. nums : Reverse of names
    8. outputLabel : Label of predicted output
    '''
    def __init__(self, labels=[], features=[], values=defaultdict(set), numExamples=0, n=0):
        self.labels = labels[:]
        self.features = features[:]
        self.values = values
        self.numExamples = numExamples
        self.n = n
        #self.m = m
        self.names = defaultdict(str)
        self.nums = defaultdict(int)
        self.outputLabel = None
        
    '''
    This function reads the data from the file named filename and 
    stores the information in the form that is easier to process 
    for the decision tree.
    '''
    def readData(self, filename):
        dataFile = open(filename, 'r')
        
        linesRead = 0
        for line in dataFile:
            linesRead += 1
            if linesRead == 1:
                self.n, self.m = map(int, line.split())
            elif linesRead == 2:
                attrs = line.split()
                self.outputLabel = attrs[0]
                for i in range(1,self.n):
                    self.names[i-1] = attrs[i]
                    self.nums[attrs[i]] = i-1
            elif linesRead == 3:
                labelType = line[0]
                types = [0 for i in range(self.n)]
                for i in range(1,self.n+1):
                    types[i-1] = line[i]
                if labelType != 'b':
                    print 'Only binary classification is allowed'
                    return 2
            else:
                line = line.split()
                self.labels.append(int(line[0]))
                self.features.append(line[1:])
                for i in range(1,self.n):
                    self.values[i-1].add(line[i])
        
        self.numExamples = linesRead - 3
        