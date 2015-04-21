import sys
sys.path.append('./Z3/build')
from z3 import *
import math
import time
import os




slv = Solver()


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
    return float(m[r].as_decimal(3).replace('?',''))

def multiplexer(indicators,choices):
    assert(len(indicators) == len(choices))
    if len(indicators) == 1:
        return choices[0]
    return If(indicators[0], choices[0],
              multiplexer(indicators[1:],choices[1:]))

# wrapper over If that handles tuples correctly
def conditional(p,q,r):
    if not isinstance(q,tuple) and not isinstance(q,list): return If(p,q,r)
    return tuple([ If(p,a,b) for a,b in zip(list(q),list(r)) ])


def compressionLoop(pr,mdl,verbose = True):
    global_start_time = time.time()
    solver_time = 0
    m = None
    while True:
        start_time = time.time()
        if verbose: print "Checking model using Z3..."
        if str(slv.check()) == 'sat':
            m = slv.model()
        else:
            break
        d = (time.time() - start_time)
        if verbose: print "Found model in %f sec" % d
        solver_time += d
        if verbose: print(pr(m))
        up = extract_real(m,mdl)
        if verbose: print "MDL", up
        slv.add(mdl < up)
        if verbose: print "Trying for a better one...\n"
    d = (time.time() - start_time)
    if verbose: print "Proved unsatisfiable in %f sec" % d
    solver_time += d
    if verbose: print "Total solver time: %f"  % solver_time
    if m == None:
        return 'FAIL',m
    return pr(m), extract_real(m,mdl)



'''
SASQUATCH
'''
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
'''
    for r in rule_bank.keys():
        if primitive_production[r]:
            print "%s is primitive" % r
'''


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
    
    indicators = pick_one(numberRules)
    
    recursive = [ [ generator(d-1, child) for child in children ]
                  for (children,printer,evaluator) in rules ]
    recursiveRun = [ [ r[0] for r in rs ] for rs in recursive]
    recursiveMDL = [ [ r[1] for r in rs ] for rs in recursive ]
    recursivePrint = [ [ r[2] for r in rs ] for rs in recursive ]
    #recursiveConcrete = [ [ r[3] for r in rs ] for rs in recursive ]
    
    childrenMDL = real()
    for i in range(numberRules):
        constrain(Implies(indicators[i],
                          childrenMDL == summation(recursiveMDL[i])))
    def printer(m):
        for i in range(numberRules):
            flag = indicators[i]
            if yes(m[flag]):
                # using the ith rule
                chosen_printer = rules[i][1]
                printed_children = [ r(m) for r in recursivePrint[i] ]
                return apply(chosen_printer,[m] + printed_children)
        return "#"+production
    
    def evaluate(i):
        outputs = []
        for r in range(numberRules):
            children_outputs = [ c(i) for c in recursiveRun[r] ]
            runner = rules[r][2]
            outputs.append(apply(runner, [i]+children_outputs))
        # multiplex the outputs; generically, they might be tuples
        if not isinstance(outputs[0],tuple):
            return multiplexer(indicators, outputs)
        output_size = len(outputs[0])
        multiplexed = []
        for j in range(output_size):
            multiplexed.append(multiplexer(indicators, [ o[j] for o in outputs ]))
        return multiplexed
    
    '''
    for j in range(numberRules):
        concrete.append((1,[indicators[j]]))
        concreteArguments = filter(lambda cs: len(cs) > 0,
                                   recursiveConcrete[j])
        
        for c in recursiveConcrete[j]:
            concrete.append([indicators[j]]+c))
    '''
    mdl = real()
    constrain(mdl == childrenMDL + logarithm(numberRules))
    return evaluate, mdl, printer #, concrete

def clear_solver():
    global slv, rule_bank
    slv = Solver()
    productions = rule_bank.keys()
    for p in productions:
        if isinstance(rule_bank[p],list):
            del rule_bank[p]

def primitive_real():
    thing = real()
    def evaluate_real(i): return thing
    def print_real(m): return extract_real(m,thing)
    return evaluate_real, 10.0, print_real
primitive_rule('REAL', primitive_real)

def primitive_Boolean():
    thing = boolean()
    def e(i): return thing
    def p(m): return str(m[thing])
    return e,1.0,p #, [(1,[thing]),(1,[Not(thing)])]
primitive_rule('BOOL', primitive_Boolean)


def primitive_angle():
    x,y = real(), real()
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



# hacking parallelism
# so many forks, it's a dinner party!!!
def dinner_party(tasks, callback, cores = 10):
    descriptors = {} # for each running process, a file descriptor
    outputs = 0 # accumulate the number of outputs
    number_tasks = len(tasks)

    while outputs < number_tasks:
        while len(tasks) > 0 and len(descriptors) < cores:
            # launch another task
            task = tasks[0]
            tasks = tasks[1:]
            
            r,w = os.pipe()
            p = os.fork()
            if p == 0:
                sys.stdout = os.fdopen(w, "w")
                print task()
                sys.exit()
            else:
                descriptors[p] = r
        # slurp some more dinner!!
        if len(descriptors) > 0:
            p,s = os.waitpid(-1,0)
            o = os.read(descriptors[p],200000)
            outputs += 1
            del descriptors[p]
            callback(o)

    return outputs

parallelBest = ""
parallelLoss = 0
def parallelCompression(pr,mdl,concrete,cores = 10):
    global parallelBest, parallelLoss
    parallelBest = ""
    parallelLoss = 0
    def make_task(k):
        def task():
            constrain(And(*k))
            return compressionLoop(pr,mdl,verbose = False)
        return task
    def continuation(solution):
        global parallelBest, parallelLoss
        loss = float(solution.split("\n")[-2])
        print "LOSS: %f, \n%s\n\n" % (loss,solution)
        if parallelBest == "" or loss < parallelLoss:
            parallelLoss = loss
            parallelBest = solution
    dinner_party([make_task(k) for k in concrete ],continuation,cores)
    print "BEST:\n%s" % parallelBest
