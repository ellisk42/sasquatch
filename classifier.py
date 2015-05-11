import random
import sys
import subprocess
import math
from find_bad_parses import *


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
            print l
        if "Test data log likelihoods" in l:
            reading_likelihoods = True
    return likelihoods

def classifier_accuracies(P,N,S):
    failures = find_bad_parses()
    positives = [ "pictures/%i_1_%i" % (P,x) for x in range(100) ]
    negatives = [ "pictures/%i_0_%i" % (P,x) for x in range(100) ]
    #positives = [p for p in positives if not (p in failures) ]
    #negatives = [p for p in negatives if not (p in failures) ]
    total_clean = len(positives) + len(negatives)
    test = ' '.join(positives + negatives)
    
    accuracies = []
    for s in range(S):
        p = random.sample(positives,N)
        n = random.sample(negatives,N)
        
        positive_likelihoods = run_output(' '.join(p),test)
        negative_likelihoods = run_output(' '.join(n),test)    
        
        correct = 0
        for j in range(total_clean):
            positive_example = j < len(positives)
            if positive_likelihoods[j] == negative_likelihoods[j]:
                correct += 0.5
            elif positive_likelihoods[j] > negative_likelihoods[j]:
                if positive_example:
                    correct += 1
            else:
                if not positive_example:
                    correct += 1
        print p,n, (correct/float(total_clean))

        accuracies.append(correct/float(total_clean))
    #len([a for a in accuracies if a > classified_threshold ])
    accuracy = sum(accuracies)/float(S)
    #    error = math.sqrt(accuracy*(1-accuracy)/float(S))
    error = variance(accuracies)
    return accuracy,math.sqrt(error/float(S))

if __name__ == "__main__":
    P = int(sys.argv[1]) # which data set are we using
    N = int(sys.argv[2]) # number of positive and negative training examples
    S = int(sys.argv[3]) # number of samples
    print classifier_accuracies(P,N,S)
