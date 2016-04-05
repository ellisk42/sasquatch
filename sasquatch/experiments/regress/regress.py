import math
from   random              import randint
from   sasquatch.sasquatch import *

def run_regression(family,testFn,nCurves=5,nSamples=4,maxDepth=3,maxParams=2):
    """ unsupervised synthesis over a family of regression problems:

    family: python expr, the family to which the sampled functions will belong
      (e.g. n1*x+n2, where n1,n2 are Real numbers)
    testFn: python expr, a specific function which may or may not be in 'family'
      but against which the learned representation will be tested (e.g. 8*x+3)
    nCurves: integer, how many specific curves to sample from family
    nSamples: integer, how many samples to generate for each sampled curve
    maxDepth: integer, the maximum depth of any considered representation's AST
    maxParams: integer, the maximum number of parameters, theta_i, to consider
    """
    costOfReal = 10.0 # hack! magic numbers
    Xs = range(nSamples)
    solutions = []
    for nParameters in range(1,maxParams+1):
        clear_solver()
        add_grammar(nParameters)
        e,modelMDL,pr = generator(maxDepth,'EXPRESSION')
        dataMDL = nCurves * nParameters * costOfReal
        totalMDL = summation([modelMDL,dataMDL])
        inputs,outputs = make_training_data(family,nParameters,nCurves,Xs)
        push_solver() # set a new frame
        constrain_accuracy(inputs,outputs,e)
        constrain_efficiency(totalMDL,solutions)
        solutions = solutions+search_for_solution(pr,e,totalMDL,nParameters)
    (solver,bestNParameters,evaluator) = print_best_solution(solutions)
    test_model(solver,bestNParameters,Xs,evaluator,testFn)

def add_grammar(nParameters):
    """ add the following grammar to the solver:
        EXPR <- REAL
              | x
              | Theta[i]
              | (+ EXPR EXPR)
              | (* EXPR EXPR)
    """
    rule('EXPRESSION', ['REAL'],
         lambda m, r: r,
         lambda i, r: r)
    rule('EXPRESSION',[],
         lambda m: 'x',
         lambda i: i[0])
    indexed_rule('EXPRESSION','Theta',
                 nParameters,
                 lambda i: i[1:])
    rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
         lambda m, p, q: "(+ %s %s)" % (p,q),
         lambda i, p, q: p+q)
    rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
         lambda m, p, q: "(* %s %s)" % (p,q),
         lambda i, p, q: p*q)

def make_training_data(family,nParams,nCurves,Xs):
    """ create the training input/output pairs """
    L = [ [ real() for i in range(nParams) ] for j in range(nCurves) ]
    training_inputs = []
    # training_inputs is a list: [[X_1 Real_1 ... Real_D], ... [X_X ...]]
    xs = [[x]+L[n] for n in range(nCurves) for x in Xs]

    f = lambda n1, n2, x: eval(family)
    thetas = [[randint(1,4),randint(1,4)] for n in range(nCurves)]
    ys = [f(thetas[n][0],thetas[n][1],x) for n in range(nCurves) for x in Xs]
    return xs,ys
        
def constrain_efficiency(totalMDL,solutions):
    """ force the next solution to be as good or better than the current best """
    if len(solutions) > 0:
        bestLength = min(solutions)[0]
        constrain(totalMDL < bestLength)
    
def constrain_accuracy(xs,ys,e,epsilon=0.1):
    """ force the next solution to be accurate """
    for x,y in zip(xs, ys):
        yp = e(x)
        constrain(yp >= y - epsilon)
        constrain(yp <= y + epsilon)

def search_for_solution(pr,e,totalMDL,D):
    """ try to find a solution and report the result """
    p,totalMDL = compressionLoop(pr,totalMDL)
    if totalMDL == None:
        print "="*40
        print "No solution for D = %i" % D
        print "="*40
        return []
    else:
        print "="*40
        print "Got solution for D = %i" % D
        print "="*40
        return [(totalMDL,p,e,pr,get_solver(),D)]
    
def print_best_solution(solutions):
    """ report the best discovered solution over all degrees of freedom """
    (m,p,e,pr,solver,D) = min(solutions)
    print
    print "="*40
    print "Best solution: %f bits (D = %i)" % (m,D)
    print "="*40
    print p
    print "="*40
    print
    return (solver,D,e)

def test_model(solver,nParams,Xs,predict,testFn):
    """ test our representation on a held out test function """
    set_solver(solver)
    thetas = [ real() for i in range(nParams)]
    for x in Xs:
        y = eval(testFn)
        yp = predict([x] + thetas)
        epsilon = 0.1
        constrain(yp > y - epsilon)
        constrain(yp < y + epsilon)

    if str(solver.check()) == 'sat': # found a satisfying model
        print
        print "="*40
        print 'Found representation for test data:'
        print "="*40
        for p in range(nParams):
            pVal = extract_real(solver.model(), thetas[p])
            print "Theta[{0:d}] = {1:.3f}".format(p, pVal)
        print "="*40
    else:
        print
        print "="*40
        print 'Test data not representable.'
        print "="*40
