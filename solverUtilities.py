import sys
import math
import time
import os

sys.path.append('./Z3/build')
from z3 import *

slv = Solver()

recent_model = None
def get_recent_model():
    global recent_model
    return recent_model

def push_solver():
    slv.push()
def pop_solver():
    slv.pop()
def set_solver(s):
    global slv
    slv = s
def get_solver():
    global slv
    return slv


next_Jensen = 1
def new_symbol():
    global next_Jensen
    next_Jensen = next_Jensen + 1
    return "gensym%i" % next_Jensen

def constrain(constraint):
    slv.add(constraint)
def yes(x):
    return str(x) == "True"

def boolean():
    return Bool(new_symbol())
def real():
    return Real(new_symbol())
def integer():
    return Int(new_symbol())

def pick_exactly_one(indicators):
    K = len(indicators)
    constrain(Or(*indicators))
    for i in range(0,K):
        for j in range(i+1,K):
            constrain(Not(And(indicators[j],indicators[i])))
def pick_one(K):
    indicators = [ boolean() for i in range(0,K) ]
    pick_exactly_one(indicators)
    return indicators
def permutation_indicators(K):
    indicators = [ [ boolean() 
                     for i in range(K) ]
                   for j in range(K) ]
    for r in range(K):
        pick_exactly_one(indicators[r])
    for c in range(K):
        pick_exactly_one([indicators[i][c] for i in range(K) ])
    return indicators
            
def booleans(K):
    return [ boolean() for i in range(K) ]
def real_numbers(K):
    return [ real() for i in range(0,K) ]
def integer_numbers(K):
    return [ integer() for i in range(0,K) ]

def summation(xs):
    accumulator = 0
    for x in xs:
        new_accumulator = real()
        constrain(new_accumulator == accumulator + x)
        accumulator = new_accumulator
    return accumulator

def logarithm(n):
    return math.log(n)/math.log(2.0)
def iff(a,b):
    constrain(And(Implies(a,b),Implies(b,a)))

def constrain_angle(dx,dy):
    constrain(dx*dx + dy*dy == 1)

def extract_real(m,r):
    if m[r] == None:
        print r
    return float(m[r].as_decimal(3).replace('?',''))
def extract_int(m,i):
    return m[i].as_long()
def extract_bool(m,b):
    b = m[b]
    if b == None:
        return '?'
    return str(b)

def multiplexer(indicators,choices):
    if isinstance(choices[0], tuple):
        n = len(choices[0]) # arity of tuple
        return tuple([ multiplexer(indicators, [x[j] for x in choices ]) for j in range(n) ])
    assert(len(indicators) == len(choices))
    if len(indicators) == 1:
        return choices[0]
    return If(indicators[0], choices[0],
              multiplexer(indicators[1:],choices[1:]))

def apply_permutation(p, xs):
    return [ multiplexer(p[j],xs) for j in range(len(xs)) ]

# wrapper over If that handles tuples correctly
def conditional(p,q,r):
    if not isinstance(q,tuple) and not isinstance(q,list): return If(p,q,r)
    return tuple([ If(p,a,b) for a,b in zip(list(q),list(r)) ])


def compressionLoop(pr,mdl,verbose = True,timeout = None,enforce_structure = True):
    global recent_model
    if timeout:
        slv.set("timeout",timeout*1000)
    global_start_time = time.time()
    solver_time = 0
    m = None
    while True:
        start_time = time.time()
        if verbose: print "Checking model using Z3..."
        if str(slv.check()) == 'sat':
            m = slv.model()
            recent_model = m
        else:
            break
        d = (time.time() - start_time)
        solver_time += d
        up = extract_real(m,mdl)
        slv.add(mdl < up)
        if verbose: 
            print "Found model in %f sec" % d
            print(pr(m))
            print "MDL", up
            print "Trying for a better one...\n"
    d = (time.time() - start_time)
    solver_time += d
    if verbose:
        print "Proved unsatisfiable in %f sec" % d
        print "Total solver time: %f"  % solver_time
    if m == None:
        return 'FAIL',m
    # compute structural assertions
    structure_constraints = []
    for v in structural_variables:
        if m[v] != None:
            structure_constraints.append(v == m[v])
    # pop the frame that contains the training data
    pop_solver()
    # constrain the solution to be the best one that we found
    if enforce_structure: slv.add(And(*structure_constraints))
    return pr(m), extract_real(m,mdl)



rule_bank = {}
dirty_bank = False
def rule(name, children, printer, evaluator):
    global rule_bank, dirty_bank
    rule_bank[name] = rule_bank.get(name,[]) + [(children,printer,evaluator)]
    dirty_bank = True
def primitive_rule(name, callback):
    global rule_bank, dirty_bank
    dirty_bank = True
    rule_bank[name] = callback

primitive_production = {}
def analyze_rule_recursion():
    global primitive_production, dirty_bank
    dirty_bank = False
    for r in rule_bank.keys():
        primitive_production[r] = not isinstance(rule_bank[r],list)
    # Go until we reach a fixed point
    changed = True
    while changed:
        changed = False
        for production in rule_bank.keys():
            if primitive_production[production]: continue
            counterexample = False
            for (children,printer,evaluator) in rule_bank[production]:
                if any([ not primitive_production[child] for child in children ]):
                    counterexample = True
                    break
            if not counterexample:
                changed = True
                primitive_production[production] = True

# structural variables are those that control the structure of the synthesized program
structural_variables = []
def structural(v):
    if isinstance(v,list):
        return [ structural(vp) for vp in v ]
    elif isinstance(v,tuple):
        return tuple([ structural(vp) for vp in v ])
    else:
        structural_variables.append(v)
        return v

def generator(d, production):
    if dirty_bank:
        analyze_rule_recursion()

    rules = rule_bank[production]
    if not isinstance(rules,list):
        return rules()

    if d <= 1 and not primitive_production[production]: # can't go any deeper
        rules = filter(lambda (children,p,e):
                           all([ primitive_production[child] for child in children]),
                       rules)

    numberRules = len(rules)

    indicators = structural(pick_one(numberRules))

    child_frequencies = {}
    for (children,printer,evaluator) in rules:
        multiset = {}
        for child in children:
            multiset[child] = 1 + multiset.get(child,0)
        for child,multiplicity in multiset.iteritems():
            child_frequencies[child] = max(multiplicity,
                                           child_frequencies.get(child,0))

    recursive = {}
    for child,multiplicity in child_frequencies.iteritems():
        for j in range(multiplicity):
            recursive["%s/%i" % (child,j+1)] = generator(d-1, child)
            
    # returns the recursive invocations for the ith rule
    def getRecursive(i, slot):
        assert slot == 0 or slot == 1 or slot == 2
        children = rules[i][0]
        child_labels = [ "%s/%i" % (child,len(filter(lambda k: k == child,children[:j]))+1)
                         for j, child in enumerate(children) ]
        return [ recursive[k][slot] for k in child_labels ]


    childrenMDL = real()
    constrain(childrenMDL == multiplexer(indicators,
                                         [ summation(getRecursive(i,1)) for i in range(numberRules)]))

    def printer(m):
        for i in range(numberRules):
            flag = indicators[i]
            if yes(m[flag]):
                # using the ith rule
                chosen_printer = rules[i][1]
                printed_children = [ r(m) for r in getRecursive(i,2) ]
                return apply(chosen_printer,[m] + printed_children)
        return "#"+production


    def evaluate(i):
        outputs = []
        for r in range(numberRules):
            children_outputs = [ c(i) for c in getRecursive(r,0) ]
            runner = rules[r][2]
            outputs.append(apply(runner, [i]+children_outputs))
        # multiplex the outputs
        return multiplexer(indicators, outputs)

    mdl = real()
    constrain(mdl == childrenMDL + logarithm(numberRules))
    return evaluate, mdl, printer #, concrete

def clear_solver():
    global slv, rule_bank, structural_variables
    structural_variables = []
    slv = Solver()
    productions = rule_bank.keys()
    for p in productions:
        if isinstance(rule_bank[p],list):
            del rule_bank[p]

def primitive_real():
    thing = structural(real())
    def evaluate_real(i): return thing
    def print_real(m): return extract_real(m,thing)
    return evaluate_real, 10.0, print_real
primitive_rule('REAL', primitive_real)

def primitive_Boolean():
    thing = structural(boolean())
    def e(i): return thing
    def p(m): return str(m[thing])
    return e,1.0,p #, [(1,[thing]),(1,[Not(thing)])]
primitive_rule('BOOL', primitive_Boolean)


def primitive_angle():
    x,y = structural((real(), real()))
    constrain_angle(x,y)
    def e(i): return x,y
    def p(m): 
        return '(angle %i)' % int(math.atan2(extract_real(m,y),extract_real(m,x))*180.0/math.pi)
    return e,10.0,p
primitive_rule('ANGLE', primitive_angle)


def indexed_rule(production, array_name, l, array_values):
    def make_index(k):
        rule(production,[],
             lambda m: "%s[%i]" % (array_name,k),
             lambda i: array_values(i)[k])
    for j in range(l):
        make_index(j)

def enum_rule(production, options):
    def make_index(k):
        rule(production,[],
             lambda m: str(options[k]),
             lambda i: options[k])
    for j in range(len(options)):
        make_index(j)

def imperative_generator(production, d, initial_production = None):
    if d == 0:
        return (lambda state,i: []),0,(lambda m: "")
    p = initial_production if initial_production else production
    first_evaluate, first_length, first_print = generator(1,p)
    
    rest_evaluate, rest_length, rest_print = imperative_generator(production, d-1)
        
    def evaluate(state,i):
        first_output, first_state = first_evaluate((state,i))
        rest_output = rest_evaluate(first_state,i)
        return [first_output]+rest_output
    def printer(m):
        return first_print(m) + "\n" + rest_print(m)
    
    mdl = real()
    constrain(mdl == first_length + rest_length)
    
    return evaluate,mdl,printer

def distribution_mode(d):
    k = {}
    for x in d:
        k[x] = k.get(x,0) + 1
    return max([ (y,x) for x,y in k.items() ])[1]
