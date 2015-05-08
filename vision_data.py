from classifier import *

N = int(sys.argv[1]) # number of positive and negative training examples
S = int(sys.argv[2]) # number of samples

averages = []
errors = []

for j in range(23):
    a,e = classifier_accuracies(j+1,N,S)
    averages.append(a)
    errors.append(e)
print averages,errors
