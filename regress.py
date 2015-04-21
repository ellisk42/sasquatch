from solverUtilities import *
import math
import sys

D = 1 # int(sys.argv[1]) # dimensionality of the encoding
S = 4 # number of samples
N = 5 # number of functions

X = range(S)
L = [ [ real() for i in range(0,D) ] for j in range(0,N) ]


# inputs is a list of [x t1 t2 ...]

rule('EXPRESSION', ['REAL'],
     lambda m, r: r,
     lambda i, r: r)
rule('EXPRESSION',[],
     lambda m: 'x',
     lambda i: i[0])
indexed_rule('EXPRESSION','T',D,
             lambda i: i[1:])
rule('EXPRESSION',['BOOL','EXPRESSION','EXPRESSION'],
     lambda m, o, p, q: "(%s %s %s)" % ('+' if o == 'True' else '*',p,q),
     lambda i, o, p, q: If(o,p+q,p*q))

training_inputs = []
training_outputs = []
for n in range(0,N):
    for x_ in X:
        training_inputs.append([x_] + L[n])
        x = x_
        training_outputs.append(eval(sys.argv[1]))

e,m,p = generator(3,'EXPRESSION')
for o,y in zip(training_inputs, training_outputs):
    epsilon = 0.1
    yp = e(o)
    constrain(yp > y - epsilon)
    constrain(yp < y + epsilon)

compressionLoop(p,m)
