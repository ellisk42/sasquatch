from solverUtilities import *
import math
import sys


S = 4 # number of samples
N = 5 # number of functions
sqN = int(math.ceil(math.sqrt(N)))

X = range(S)

training_outputs = []
for n in range(1,N+1):
    n1 = 3*(int(n/sqN) + 1)
    n2 = 3*(n % sqN + 1)
    for x in X:
        training_outputs.append(eval(sys.argv[1]))

solutions = []
for D in range(1,3):
    dataMDL = N*D*10.0
    clear_solver()
    # inputs is a list of [x t1 t2 ...]
    L = [ [ real() for i in range(0,D) ] for j in range(0,N) ]
    
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
    for n in range(N):
        for x in X:
            training_inputs.append([x] + L[n])
    e,m,p = generator(3,'EXPRESSION')
    m = summation([m,dataMDL])
    for o,y in zip(training_inputs, training_outputs):
        epsilon = 0.1
        yp = e(o)
        constrain(yp > y - epsilon)
        constrain(yp < y + epsilon)
    p,m = compressionLoop(p,m)
    if m == None:
        print "No solution for D = %i" % D
    else:
        print "Got solution for D = %i" % D
        solutions.append((m,p,D))
        

(m,p,D) = min(solutions)
print "="*40
print "Best solution: %f bits (D = %i)" % (m,D)
print "="*40

print p
