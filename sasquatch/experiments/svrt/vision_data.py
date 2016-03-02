from classifier import *
from dinnerParty import *
import numpy as np


N = int(sys.argv[1]) # number of positive and negative training examples
S = int(sys.argv[2]) # number of samples


def task(t):
    return Bayesian_classifier(t,N,1)
    
def flatten(xs):
    return [i for l in xs for i in l  ]

T = 3

arguments = flatten([[t]*S for t in range(1,T+1) ])

a = parallel_map(task, arguments, cores = 35)
a = flatten([eval(x) for x in a ])

a = np.array(a).reshape((T,len(a)/T)).tolist()
print a
