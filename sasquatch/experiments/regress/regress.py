import math
import sys
from   sasquatch import *

def run_regression(trainingFn,testingFn):
    nSamples = 4
    N = 5 # number of functions?
    X = range(nSamples) # What is X?
    solutions = []
    for D in range(1,3): # Dimensionality?
        clear_solver()
        add_grammar_to_solver(D)
        e,modelMDL,pr = generator(3,'EXPRESSION') # evaluate w/ depth = 3
        dataMDL = N*D*10.0 
        totalMDL = summation([modelMDL,dataMDL])
        training_inputs = create_training_inputs(D,N,X)
        training_outputs = create_training_outputs(trainingFn,N,X)
        push_solver() # set a new frame for training data constraints
        constrain_approximate_correctness(training_inputs,training_outputs,e)
        constrain_mdl_based_on_search(totalMDL,solutions)
        solutions = solutions+search_for_solution(pr,e,totalMDL,D)
    print_best_solution(solutions)
#    test_model(solver,D,X,e)

def add_grammar_to_solver(D):
    """ add the following grammar to the solver:
        EXPR <- REAL
              | x
              | T[i]
              | (+ EXPR EXPR)
              | (* EXPR EXPR)
    """
    rule('EXPRESSION', ['REAL'],
         lambda m, r: r,
         lambda i, r: r)
    rule('EXPRESSION',[],
         lambda m: 'x',
         lambda i: i[0])
    indexed_rule('EXPRESSION','T',
                 D,
                 lambda i: i[1:])
    rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
         lambda m, p, q: "(+ %s %s)" % (p,q),
         lambda i, p, q: p+q)
    rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
         lambda m, p, q: "(* %s %s)" % (p,q),
         lambda i, p, q: p*q)

def search_for_solution(pr,e,totalMDL,D):
    p,totalMDL = compressionLoop(pr,totalMDL)
    if totalMDL == None:
        print "No solution for D = %i" % D
        return []
    else:
        print "Got solution for D = %i" % D
        return [(totalMDL,p,e,pr,get_solver(),D)]
    
def constrain_mdl_based_on_search(totalMDL,solutions):
    if len(solutions) > 0:
        bestLength = min(solutions)[0]
        constrain(totalMDL < bestLength)
    
def print_best_solution(solutions):
    (m,p,e,pr,solver,D) = min(solutions)
    print "="*40
    print "Best solution: %f bits (D = %i)" % (m,D)
    print "="*40
    print p
    print "="*40

def constrain_approximate_correctness(xs,ys,e,epsilon=0.1):
    for x,y in zip(xs, ys):
        yp = e(x)
        constrain(yp >= y - epsilon)
        constrain(yp <= y + epsilon)

def create_training_inputs(D,N,X):
    # training_inputs is a list: [[X_1 Real_1 ... Real_D], ... [X_X ...]]
    L = [ [ real() for i in range(D) ] for j in range(N) ]
    training_inputs = []
    for n in range(N): # for every function
        for x in X: # for every sample
            training_inputs.append([x] + L[n])
    return training_inputs

def create_training_outputs(trainingFn,N,X):
    # create the outputs by evaluating the functions
    training_outputs = []
    sqN = int(math.ceil(math.sqrt(N)))
    for n in range(1,N+1): # for every function
        n1 = 3*(int(n/sqN) + 1)
        n2 = 3*(n % sqN + 1)
        for x in X: # for every sample
            training_outputs.append(eval(trainingFn))
    print training_outputs
    return training_outputs

def test_model(solver,D,X,e):
    # Try it on a held out test function
    set_solver(solver)
    t = [ real() for i in  range(D)]
    for x in X:
        y = eval(testingFn)
        yp = e([x] + t)
        epsilon = 0.1
        # constrain(yp ~= y)
        constrain(yp > y - epsilon)
        constrain(yp < y + epsilon)

    # if the test function works, print that, too
    if str(solver.check()) == 'sat':
        print 'Found representation for test data:'
        print t
        print [solver.model()[th] for th in t]
        # print [extract_real(solver.model(), th) for th in t ]
    else:
        print 'Test data not representable.'
