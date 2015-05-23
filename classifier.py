import random
import sys
import subprocess
import math
from find_bad_parses import *
import re

VERBOSE = False

def variance(xs):
    a = sum(xs)/float(len(xs))
    return sum([(x-a)**2 for x in xs ])


def run_output(training,test = ''):
    if isinstance(training,list):
        training = ' '.join(training)
    if len(test) > 0:
        command = "python vision.py --quiet %s %s %s" % (training, "test" if test else '',test)
    else:
        command = "python vision.py --quiet %s" % training
    o = subprocess.check_output(command, 
                                shell = True)
    likelihoods = []
    reading_likelihoods = False
    ls = o.split('\n')
    marginal_likelihood = None
    
    for l in ls:
        if len(l) == 0: continue
        m = re.match("Marginal likelihood: (.+)$",l)
        if m:
            marginal_likelihood = m.group(1)
            if "infinity" in marginal_likelihood:
                marginal_likelihood = -float('inf')
            else:
                marginal_likelihood = -float(marginal_likelihood)
        if reading_likelihoods:
            likelihoods.append(float(l))
        else:
            if VERBOSE:
                print l
        if "Test data log likelihoods" in l:
            reading_likelihoods = True

    assert marginal_likelihood != None
    if len(test) > 0:
        return likelihoods
    else:
        return marginal_likelihood

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
            print ' '.join(p),'\t',' '.join(n), '\t',(correct/float(test_size))

        accuracies.append(correct/float(test_size))
    accuracy = sum(accuracies)/float(S)
    error = variance(accuracies)
    return accuracies
#    return accuracy,math.sqrt(error/float(S))

def Bayesian_classifier(P,N,S):
    positives = [ "pictures/%i_1_%i" % (P,x) for x in range(100) ]
    negatives = [ "pictures/%i_0_%i" % (P,x) for x in range(100) ]
    
    accuracies = []
    for s in range(S):
        p = random.sample(positives,N+1)
        n = random.sample(negatives,N+1)
        # class a, class B, test a, test be
        class_a = p[1:]
        class_B = n[1:]
        test_a = p[0]
        test_be = n[0]
        
        # likelihood of each cluster on its own
        likelihood_a = run_output(class_a)
        likelihood_be = run_output(class_B)
        print "A,B = ",likelihood_a,likelihood_be
        
        # likelihood_of_each_cluster with a test point added
        a_test_b = run_output(class_a + [test_be])
        a_test_a = run_output(class_a + [test_a])
        b_test_b = run_output(class_B + [test_be])
        b_test_a = run_output(class_B + [test_a])
        print "A+a = ",a_test_a
        print "A+b = ",a_test_b
        print "B+a = ",b_test_a
        print "B+b = ",b_test_b
        
        # how did we do with test a?
        marginal_a = a_test_a + likelihood_be
        marginal_b = b_test_a + likelihood_a
        if marginal_a == marginal_b:
            accuracies.append(0.5)
        elif marginal_a > marginal_b:
            accuracies.append(1)
        else:
            accuracies.append(0)
            
        # how did we do with test b?
        marginal_a = a_test_b + likelihood_be
        marginal_b = b_test_b + likelihood_a
        if marginal_a == marginal_b:
            accuracies.append(0.5)
        elif marginal_a > marginal_b:
            accuracies.append(0)
        else:
            accuracies.append(1)
    return accuracies
        

if __name__ == "__main__":
    VERBOSE = True
    P = int(sys.argv[1]) # which data set are we using
    N = int(sys.argv[2]) # number of positive and negative training examples
    S = int(sys.argv[3]) # number of samples
#    print classifier_accuracies(P,N,S)
    print Bayesian_classifier(P,N,S)

    
