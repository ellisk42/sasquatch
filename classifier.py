import random
import sys
import subprocess


P = int(sys.argv[1]) # which data set are we using
N = int(sys.argv[2]) # number of positive and negative training examples
S = int(sys.argv[3]) # number of samples

positives = [ "pictures/%i_1_%i" % (P,x) for x in range(100) ]
negatives = [ "pictures/%i_0_%i" % (P,x) for x in range(100) ]
test = ' '.join(positives + negatives)




def run_output(training):
    o = subprocess.check_output("python vision.py %s test %s" % (training, test), shell = True)
    likelihoods = []
    reading_likelihoods = False
    for l in o.split('\n'):
        if len(l) == 0: continue
        if reading_likelihoods:
            likelihoods.append(float(l))
        if "Test data log likelihoods" in l:
            reading_likelihoods = True
    return likelihoods[:100], likelihoods[100:]

accuracies = []
for s in range(S):
    p = random.sample(positives,N)
    n = random.sample(negatives,N)
    
    pp,pn = run_output(' '.join(p))
    np,nn = run_output(' '.join(n))    
    
    correct = 0
    for j in range(100):
        # positive example j
        if pp[j] > np[j]:
            correct += 1
        elif pp[j] == np[j]:
            correct += 0.5
        # negative example j
        if nn[j] > pn[j]:
            correct += 1
        elif nn[j] == pn[j]:
            correct += 0.5
    print p,n, correct
    accuracies.append(correct/200.0)
print accuracies
average = sum(accuracies)/float(S)
variance = sum([(accuracy - average)**2 for accuracy in accuracies ])/float(S)
print average
print variance
