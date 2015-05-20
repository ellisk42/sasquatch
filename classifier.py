import random
import sys
import subprocess
import math
from find_bad_parses import *


VERBOSE = False
# need to have at least this accuracy to count as solving the task
classified_threshold = 0.9

def variance(xs):
    a = sum(xs)/float(len(xs))
    return sum([(x-a)**2 for x in xs ])

def run_output(training,test):
    o = subprocess.check_output("python vision.py %s test %s" % (training, test), shell = True)
    likelihoods = []
    reading_likelihoods = False
    for l in o.split('\n'):
        if len(l) == 0: continue
        if reading_likelihoods:
            likelihoods.append(float(l))
        else:
            if VERBOSE:
                print l
        if "Test data log likelihoods" in l:
            reading_likelihoods = True
    return likelihoods

def classifier_accuracies(P,N,S):
    failures = find_bad_parses()
    positives = [ "pictures/%i_1_%i" % (P,x) for x in range(100) ]
    negatives = [ "pictures/%i_0_%i" % (P,x) for x in range(100) ]
    total_clean = len(positives) + len(negatives)
    
    accuracies = []
    for s in range(S):
        p = random.sample(positives,N)
        n = random.sample(negatives,N)
        training = p+n
        
        positives_test = [ h for h in positives if (not (h in training)) ]
        negatives_test = [ h for h in negatives if (not (h in training)) ]
        test = ' '.join(positives_test + negatives_test)        
        
        positive_likelihoods = run_output(' '.join(p),test)
        negative_likelihoods = run_output(' '.join(n),test)    
        
        correct = 0
        test_size = len(positives_test) + len(negatives_test)
        for j in range(test_size):
            positive_example = j < len(positives_test)
            if positive_likelihoods[j] == negative_likelihoods[j]:
                correct += 0.5
            elif positive_likelihoods[j] > negative_likelihoods[j]:
                if positive_example:
                    correct += 1
            else:
                if not positive_example:
                    correct += 1
        if VERBOSE:
            print p,n, (correct/float(test_size))

        accuracies.append(correct/float(test_size))
    accuracy = sum(accuracies)/float(S)
    error = variance(accuracies)
    return accuracies
#    return accuracy,math.sqrt(error/float(S))

if __name__ == "__main__":
    VERBOSE = True
    P = int(sys.argv[1]) # which data set are we using
    N = int(sys.argv[2]) # number of positive and negative training examples
    S = int(sys.argv[3]) # number of samples
    print classifier_accuracies(P,N,S)

    
