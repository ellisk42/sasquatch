from classifier import *
from dinnerParty import *

N = int(sys.argv[1]) # number of positive and negative training examples
S = int(sys.argv[2]) # number of samples

averages = []
errors = []

def task(t):
    return Bayesian_classifier(t,N,1)
    


T = 23

arguments = [[t]*S for t in range(1,T+1) ]
arguments = [i for l in arguments for i in l  ]

a = parallel_map(task, arguments, cores = T)
a = [eval(x) for x in a ]
a = np.array(a).reshape((T,S)).tolist()
print a
