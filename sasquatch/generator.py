"""construct programs according to some representation language"""

from .utilities   import *
from .constraints import *
from .language    import Language,is_primitive
from itertools    import chain
from z3           import Sum

def count_children(rules):
    child_frequencies = {}
    for (children,printer,evaluator,cs) in rules:
        multiset = {}
        for child in children:
            multiset[child] = 1 + multiset.get(child,0)
        for child,multiplicity in multiset.iteritems():
            child_frequencies[child] = max(multiplicity,
                                           child_frequencies.get(child,0))
    return child_frequencies

def make_getRecursive(gen,rules,d):
    # recursively call generator for each possible call to each child
    recursive = {"{0:s}/{1:d}".format(child,j+1) : gen.generate(child,d-1)
                 for child,multiplicity in count_children(rules).iteritems()
                 for j in range(multiplicity)}

    def getRecursive(i, slot):
        """return the requested information for each child of rule[i]"""
        assert slot in [0,1,2] # i.e. evaluate,mdl,show
        children = rules[i][0]
        # figure out the child multiplicity of each child of rule[i]
        child_labels = [ "%s/%i" % (child,len(filter(lambda k: k == child,
                                                     children[:j])) + 1)
                         for j, child in enumerate(children) ]
        return [ recursive[k][slot] for k in child_labels ]
    return getRecursive

class Generator(object):
    def __init__(self,values,lang=None):
        self.structural_variables = []
        self.constraints = []
        self.values = values
        self.lang = lang if lang else Language(values)

    def mark_as_structural(self,variable):
        """updating the structural variables for a program"""
        if is_a(variable,list,tuple):
            v = [ self.mark_as_structural(v) for v in variable ]
        else:
            self.structural_variables.append(variable)
            v = variable
        return tuple(v) if is_a(variable,tuple) else v

    def generate_primitive(self,rules):
        ev,mdl,sh,cs = rules()
        self.mark_as_structural(ev(None)) # ev(None) return the 'thing'
        self.constraints += cs
        return ev,mdl,sh

    def mindepth_prod(self,production,d):
        """is a production's minimum AST depth <= d?

        a production's minimum depth is 0 if a primitive production else
        the minimum depth of that production's rules

        """
        rules = self.lang.rule_bank[production]
        return (d >= 0 and
                (self.mindepth_prod(production,d-1) or
                 is_primitive(rules) or
                 any([self.mindepth_rule(r,d) for r in rules])))

    def mindepth_rule(self,(cs,e,p,xs),d):
        """is a rule's minimum AST depth <= d?

        a rule's minimum depth is 1 + the max child production mindepth

        """
        return (d >= 0 and
                (self.mindepth_rule((cs,e,p,xs),d-1) or
                 all([self.mindepth_prod(c,d-1) for c in cs])))

    def generate(self,production, d):
        """provide a tuple to evaluate, score, and print a production"""
        rules = self.lang.rule_bank[production]

        if is_primitive(rules):
            return self.generate_primitive(rules) # adds constraints set by primitives

        productionMDL = logarithm(len(rules))

        # keep only those rules we can produce in the remaining depth
        # FIXME: cache/memoize for speed
        rules = [r for r in rules if self.mindepth_rule(r,d)]

        getRecursive = make_getRecursive(self,rules,d)

        numberRules = len(rules)

        indicators = self.mark_as_structural(self.values("b",numberRules))
        self.constraints += pick_exactly_one(indicators)

        childrenMDL = self.values("r")
        outputs,cs = zip(*[ summation(self.values,getRecursive(i,1))
                            for i in range(numberRules) ])
        self.constraints.append(childrenMDL == multiplexer(indicators,outputs))
        self.constraints += list(chain.from_iterable(cs))

        def evaluate(i):
            """evaluate each possibility, return the outputs, force a choice"""
            outputs = []
            for r in range(numberRules):
                children_outputs = [ c(i) for c in getRecursive(r,0) ]
                runner = rules[r][2]
                outputs.append(apply(runner, [i]+children_outputs))
            return multiplexer(indicators, outputs)

        mdl = self.values("r")
        self.constraints.append(mdl == childrenMDL + productionMDL)

        def show(m):
            """convert 'production' to string under model 'm'"""
            for i in range(numberRules):
                flag = indicators[i]
                if yes(m[flag]):
                    chosen_printer = rules[i][1]
                    printed_children = [ r(m) for r in getRecursive(i,2) ]
                    return apply(chosen_printer,[m] + printed_children)
            return "#"+production

        return evaluate, mdl, show

    # FIXME: Is there a bug here such that we should be generating
    # based on the children of the initial generation? I'm not quite
    # sure what being "imperative" accomplishes for us here.
    def generate_imperative(self,production, d, initial_production = None):
        """provide a tuple that evaluates, scores, and prints a production"""
        if d == 0: # can't do any more work
            return (lambda state,i: []),0,(lambda m: "")

        # Generate programs of depth 1 resolving to p
        p = initial_production if initial_production else production
        firstEvaluate, firstMDL, firstShow, firstConstraints = self.generate(p,1)

        # generate programs of depth d-1 resolving to production
        restEvaluate, restMDL, restShow, restConstraints  = \
            self.generate_imperative(production, d-1)

        def evaluate(state,i):
            first_output, first_state = firstEvaluate((state,i))
            rest_output = restEvaluate(first_state,i)
            return [first_output]+rest_output

        mdl = self.values("r")

        def show(m):
            return firstShow(m) + "\n" + restShow(m)

        self.constraints += firstConstraints
        self.constraints += restConstraints
        self.constraints.append(mdl == Sum(firstMDL, restMDL))

        return evaluate,mdl,show
