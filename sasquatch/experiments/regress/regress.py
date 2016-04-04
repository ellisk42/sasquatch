import math
from   random                import randint
from   sasquatch.constraints import *
from   sasquatch.values      import valueMaker,extract_real
from   sasquatch.language    import Language
from   sasquatch.sasquatch   import Sasquatch

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
    values = valueMaker()
    Xs = range(nSamples)
    solutions = []
    for nParameters in range(maxParams+1):
        print "using {0} parameters".format(nParameters)
        s = Sasquatch(values,'test',verbose=True)
        ev,modelMDL,pr = s.make_search_space(build_grammar(values,nParameters),
                                             'EXPRESSION',maxDepth)
        dataMDL = nCurves * nParameters * costOfReal
        inputs,outputs = make_training_data(values,family,nParameters,nCurves,Xs)
        s.accuracy_constraints = accuracy(inputs,outputs,ev)
        totalMDL,s.efficiency_constraints = efficiency(values,
                                                       [dataMDL,modelMDL],
                                                       solutions)
        solutions = solutions+search_for_solution(s,pr,ev,totalMDL,nParameters)
    (s,bestNParameters,evaluator) = print_best_solution(solutions)
    test_model(s,values,bestNParameters,Xs,evaluator,testFn)

def build_grammar(values,nParameters):
    """ add the following grammar to the solver:
        EXPR <- REAL
              | x
              | Theta[i]
              | (+ EXPR EXPR)
              | (* EXPR EXPR)
    """
    l = Language(values)
    l.add_reals()
    l.rule('EXPRESSION', ['REAL'],
           lambda m, r: r,
           lambda i, r: r,
           [])
    l.rule('EXPRESSION',[],
           lambda m: 'x',
           lambda i: i[0],
           [])
    l.indexed_rule('EXPRESSION','Theta',
                   nParameters,
                   lambda i: i[1:])
    l.rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
           lambda m, p, q: "(+ %s %s)" % (p,q),
           lambda i, p, q: p+q,
           [])
    l.rule('EXPRESSION',['EXPRESSION','EXPRESSION'],
           lambda m, p, q: "(* %s %s)" % (p,q),
           lambda i, p, q: p*q,
           [])
    return l

def make_training_data(values,family,nParams,nCurves,Xs):
    """ create the training input/output pairs """
    L = [ [ values('r') for i in range(nParams) ] for j in range(nCurves) ]
    training_inputs = []
    # training_inputs is a list: [[X_1 Theta[0] ... Theta[nParams-1]], ...]
    xs = [[x]+L[n] for n in range(nCurves) for x in Xs]

    f = lambda n1, n2, x: eval(family)
    thetas = [[randint(1,4),randint(1,4)] for n in range(nCurves)]
    ys = [f(thetas[n][0],thetas[n][1],x) for n in range(nCurves) for x in Xs]
    return xs,ys
        
def efficiency(values,mdls,solutions):
    """ force the next solution to be as good or better than the current best """
    totalMDL,cs = summation(values,mdls)
    constraints = cs + ([totalMDL < min(solutions)[0]] if solutions else [])
    return  totalMDL,constraints
    
def accuracy(xs,ys,e,epsilon=0.1):
    """ force the next solution to be accurate """
    constraints = []
    for x,y in zip(xs, ys):
        yp = e(x)
        constraints += [yp >= y - epsilon, yp <= y + epsilon]
    return constraints

def search_for_solution(s,pr,e,totalMDL,D):
    """ try to find a solution and report the result """
    p,totalMDL = s.compress(pr,totalMDL)
    if totalMDL == None:
        print "="*40
        print "No solution for D = %i" % D
        print "="*40
        return []
    else:
        print "="*40
        print "Got solution for D = %i" % D
        print "="*40
        return [(totalMDL,p,e,pr,s,D)]
    
def print_best_solution(solutions):
    """ report the best discovered solution over all degrees of freedom """
    (m,p,e,pr,solver,D) = min(solutions)
    print
    print "="*40
    print "Best solution: %f bits (D = %i)" % (m,D)
    print "="*40
    print p
    print "="*40
    return (solver,D,e)

def test_model(s,values,nParams,Xs,predict,testFn):
    """ test our representation on a held out test function """
    thetas = [ values('r') for i in range(nParams)]
    for x in Xs:
        y = eval(testFn)
        yp = predict([x] + thetas)
        epsilon = 0.1
        s.constrain([yp >= y - epsilon, yp <= y + epsilon])

    if str(s.slv.check()) == 'sat': # found a satisfying model
        print
        print "="*40
        print 'Found representation for test data:'
        print "="*40
        for p in range(nParams):
            pVal = extract_real(s.slv.model(), thetas[p])
            print "Theta[{0:d}] = {1:.3f}".format(p, pVal)
        print "="*40
    else:
        print
        print "="*40
        print 'Test data not representable.'
        print "="*40
