import sys
import math
import time
from   z3   import *

# global variables
slv = Solver()
recent_model = None
recent_total_time = None
next_Jensen = 1
rule_bank = {}
dirty_bank = False
primitive_production = {}
structural_variables = []

# defs

## true utility functions
def new_symbol():
    """create a new generic symbol with a unique name"""
    global next_Jensen
    next_Jensen = next_Jensen + 1
    return "gensym%i" % next_Jensen

def yes(x):
    """check if a value represents 'True'"""
    return str(x) == "True"

def logarithm(n):
    """base 2 logarithm"""
    return math.log(n)/math.log(2.0)

def distribution_mode(d):
    """give the most frequent item in some container"""
    k = {}
    for x in d:
        k[x] = k.get(x,0) + 1
    return max([ (y,x) for x,y in k.items() ])[1]

## manipulate the solver
def push_solver():
    """add a backtracking point"""
    slv.push()

def pop_solver():
    """backtrack to most recent backtracking point"""
    slv.pop()

def set_solver(s):
    """change the global solver"""
    global slv
    slv = s

def get_solver():
    """get the global solver"""
    global slv
    return slv

def clear_solver():
    """reset the solver, rule bank, and structural variables"""
    global slv, rule_bank, structural_variables
    structural_variables = []
    slv = Solver()
    productions = rule_bank.keys()
    for p in productions:
        if isinstance(rule_bank[p],list):
            del rule_bank[p]

## create individual variables
def boolean():
    """create a uniquely named Boolean"""
    return Bool(new_symbol())

def real():
    """create a uniquely named Real"""
    return Real(new_symbol())

def integer():
    """create a uniquely named Integer"""
    return Int(new_symbol())

## create lists of variables
def booleans(K):
    """create K new booleans in a list"""
    return [ boolean() for i in range(0,K) ]

def real_numbers(K):
    """create K new reals in a list"""
    return [ real() for i in range(0,K) ]

def integer_numbers(K):
    """create K new integers in a list"""
    return [ integer() for i in range(0,K) ]

## add constraints
def constrain(constraint):
    """add a constraint to the solver"""
    slv.add(constraint)

def pick_exactly_one(indicators):
    """add constraints to choose exactly one of several options"""
    K = len(indicators)
    constrain(Or(*indicators))
    for i in range(0,K):
        for j in range(i+1,K):
            constrain(Not(And(indicators[j],indicators[i])))
    return indicators

def permutation_indicators(K):
    """not sure this is actually a permutation, but will have to look at
    how it is used
    """
    indicators = [ [ boolean() 
                     for i in range(K) ]
                   for j in range(K) ]
    for r in range(K):
        pick_exactly_one(indicators[r])
    for c in range(K):
        pick_exactly_one([indicators[i][c] for i in range(K) ])
    return indicators

def multiplexer(indicators,choices):
    """creates a constraint acting as a case analysis pairing indicators and choices"""
    if isinstance(choices[0], tuple):
        n = len(choices[0]) # arity of tuple
        return tuple([ multiplexer(indicators, [x[j] for x in choices ])
                       for j in range(n) ])
    assert(len(indicators) == len(choices))
    if len(indicators) == 1:
        return choices[0]
    return If(indicators[0], choices[0],
              multiplexer(indicators[1:],choices[1:]))

def apply_permutation(p, xs):
    """construct a list of multiplexer using a list of lists of indicators"""
    return [ multiplexer(p[j],xs) for j in range(len(xs)) ]

def summation(xs):
    """constrain a cumulative sum"""
    accumulator = 0
    for x in xs:
        new_accumulator = real()
        constrain(new_accumulator == accumulator + x)
        accumulator = new_accumulator
    return accumulator

def iff(a,b):
    """add bi-directional implication constraints"""
    constrain(And(Implies(a,b),Implies(b,a)))

def constrain_angle(dx,dy):
    """constrain dx,dy to be on the unit circle"""
    constrain(dx*dx + dy*dy == 1)

def conditional(p,q,r):
    """wrapper over If that handles tuples correctly"""
    if not isinstance(q,tuple) and not isinstance(q,list): return If(p,q,r)
    return tuple([ If(p,a,b) for a,b in zip(list(q),list(r)) ])

## convert Solver variables to Python values
def extract_real(m,r):
    """pulls a float from a Z3 variable, returns a float rather than a string"""
    return '?' if (m[r] == None) else float(m[r].as_decimal(3).replace('?',''))

def extract_int(m,i):
    """pulls a string representation of a Z3 variable as an int"""
    return '?' if (m[i] == None) else m[i].as_long()

def extract_bool(m,b):
    """pulls a string representation of a Z3 variable as a Boolean"""
    return '?' if (m[b] == None) else str(m[b])

## learning model
def get_recent_model():
    """get the current recent model"""
    global recent_model
    return recent_model

def get_recent_total_time():
    """get the recent solver time"""
    global recent_total_time
    return recent_total_time

def compressionLoop(pr,mdl,verbose = True,timeout = None,enforce_structure = True):
    """iteratively search for shorter descriptions of the training data"""
    global recent_model,recent_total_time

    # set the global cost function value
    constrain(Real("global_cost_function") == mdl)
    # write out the current solver to disk
    with open('.'.join(sys.argv) + '.smt2','w') as f:
        f.write(slv.to_smt2())
    # set the solver timeout, if any
    if timeout:
        slv.set("timeout",timeout*1000)
    # iteratively look for better models until the solver no longer gives 'SAT'
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
        if verbose: # optionally print a few details of the model
            print "Found model in %f sec" % d
            print(pr(m))
            print "MDL", up
            print "Trying for a better one...\n"
    # update some time-keeping details
    d = (time.time() - start_time)
    solver_time += d
    recent_total_time = solver_time
    if verbose: # optionally print a few details
        print "Proved unsatisfiable in %f sec" % d
        print "Total solver time: %f"  % solver_time
    if m == None: # return the failed model if you cannot beat the suggested mdl
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

## create rules for what Sasquatch will use as the representation language
def rule(name, children, printer, evaluator):
    """add an arbitrary rule to the rule bank"""
    global rule_bank, dirty_bank
    rule_bank[name] = rule_bank.get(name,[]) + [(children,printer,evaluator)]
    dirty_bank = True
    
def primitive_rule(name, callback):
    """add rules governing primitive values to the rule bank"""
    global rule_bank, dirty_bank
    dirty_bank = True
    rule_bank[name] = callback

def primitive_real():
    """gives a tuple representing a real in our representation language"""
    thing = structural(real())
    mdl = 10.0
    def eval_real(i): return thing
    def print_real(m): return extract_real(m,thing)
    return eval_real,mdl,print_real

def primitive_Boolean():
    """gives a tuple representing a bool in our representation language"""
    thing = structural(boolean())
    mdl = 1.0
    def eval_bool(i): return thing
    def print_bool(m): return str(m[thing])
    return eval_bool,mdl,print_bool

def primitive_angle():
    """gives a tuple representing an angle in our representation langauge"""
    x,y = structural((real(), real()))
    constrain_angle(x,y)
    mdl = 10.0
    def eval_angle(i): return x,y
    def print_angle(m):
        return '(angle %i)' % int(math.atan2(extract_real(m,y),
                                             extract_real(m,x)) *
                                  180.0/math.pi)
    return eval_angle,mdl,print_angle

def indexed_rule(production, array_name, l, array_values):
    """add rules for indexing into an array (with mutable values?)"""
    def make_index(k):
        rule(production,[],
             lambda m: "%s[%i]" % (array_name,k),
             lambda i: array_values(i)[k])
    for j in range(l):
        make_index(j)

def enum_rule(production, options):
    """add rules for indexing into an enumeration"""
    def make_index(k):
        rule(production,[],
             lambda m: str(options[k]),
             lambda i: options[k])
    for j in range(len(options)):
        make_index(j)

def analyze_rule_recursion():
    """updates 'primitive_production' so that
       'primitive_production[prod_name]' indicates whether 'prod_name'
       is ultimately a combination of primitives or is recursive (no
       other options, such as algebraic data types?)
    """
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

def structural(v):
    """structural variables := variables controling the structure of the
    synthesized program
    """
    if isinstance(v,list):
        return [ structural(vp) for vp in v ]
    elif isinstance(v,tuple):
        return tuple([ structural(vp) for vp in v ])
    else:
        structural_variables.append(v)
        return v

def generator(d, production):
    """provide a tuple to evaluate, score, and print a production"""
    # do we need a global declaration for dirty_bank and rule_bank here?
    if dirty_bank:
        analyze_rule_recursion()

    # if we have a primitive production, just return that
    rules = rule_bank[production]
    if not isinstance(rules,list):
        return rules()

    # if we can't go any deeper, limit ourselves to just the primitive productions
    if d <= 1 and not primitive_production[production]:
        rules = filter(lambda (children,p,e):
                           all([ primitive_production[child] for child in children]),
                       rules)

    # count the most frequent occurrence of each child across all possible rules
    child_frequencies = {}
    for (children,printer,evaluator) in rules:
        multiset = {}
        for child in children:
            multiset[child] = 1 + multiset.get(child,0)
        for child,multiplicity in multiset.iteritems():
            child_frequencies[child] = max(multiplicity,
                                           child_frequencies.get(child,0))

    # recursively call generator for each possible call to each child
    recursive = {}
    for child,multiplicity in child_frequencies.iteritems():
        for j in range(multiplicity):
            recursive["%s/%i" % (child,j+1)] = generator(d-1, child)
            
    def getRecursive(i, slot):
        """return the requested portion of the generators for each child in rule[i]"""
        assert slot == 0 or slot == 1 or slot == 2 # i.e. evaluate,mdl,print
        children = rules[i][0]
        # figure out what the child multiplicity is each child of rule[i]
        child_labels = [ "%s/%i" % (child,len(filter(lambda k: k == child,children[:j]))+1)
                         for j, child in enumerate(children) ]
        return [ recursive[k][slot] for k in child_labels ]

    # figure out how many rules we have to consider
    numberRules = len(rules)
    # restrict ourselves to choosing just one rule
    indicators = structural(pick_exactly_one(booleans(numberRules)))
    # setup a new variable representing the MDL of the production's children
    childrenMDL = real()
    # associate our rule choice with ???
    constrain(childrenMDL == multiplexer(indicators,
                                         [ summation(getRecursive(i,1)) for i in range(numberRules)]))

    def printer(m):
        """print 'production' under model 'm'"""
        for i in range(numberRules):
            flag = indicators[i]
            if yes(m[flag]): # using the ith rule
                chosen_printer = rules[i][1]
                printed_children = [ r(m) for r in getRecursive(i,2) ]
                return apply(chosen_printer,[m] + printed_children)
        return "#"+production

    def evaluate(i):
        # evaluate each possible choice, and return them all with the restriction that only one can actually occur
        outputs = []
        for r in range(numberRules):
            children_outputs = [ c(i) for c in getRecursive(r,0) ]
            runner = rules[r][2]
            outputs.append(apply(runner, [i]+children_outputs))
        # multiplex the outputs so we get the right output based on the chosen sub-production
        return multiplexer(indicators, outputs)

    # the complete MDL is the MDL of the children + the entropy of the rule choice
    mdl = real()
    constrain(mdl == childrenMDL + logarithm(numberRules))
    return evaluate, mdl, printer

def imperative_generator(production, d, initial_production = None):
    """provide a tuple that evaluates, scores, and prints a production"""
    if d == 0:    # can't do any more work
        # FIXME: should we declare mdl as a variable and constrain it to 0?
        return (lambda state,i: []),0,(lambda m: "")

    # not clear on what's happening here. We might evaluate one step of the production
    p = initial_production if initial_production else production
    first_evaluate, first_length, first_print = generator(1,p)

    # then we evaluate d-1 steps of that same production, without modification?
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

# non-defs
primitive_rule('REAL', primitive_real)
primitive_rule('BOOL', primitive_Boolean)
primitive_rule('ANGLE', primitive_angle)
