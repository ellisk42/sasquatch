"""high-level constraints on the structure of Sasquatch programs"""

from .utilities import is_a,is_not_a
from z3         import If,Sum,And,Or,Not,Implies
from itertools  import chain

def multiplexer(indicators,choices):
    """return a case analysis pairing boolean indicators and choices"""
    if is_a(choices[0], tuple):
        n = len(choices[0]) # arity of tuple
        return tuple([ multiplexer(indicators, [x[j] for x in choices ])
                       for j in range(n) ])
    assert(len(indicators) == len(choices))
    if len(indicators) == 1:
        return choices[0]
    return If(indicators[0], choices[0],
              multiplexer(indicators[1:],choices[1:]))

def conditional(p,q,r):
    """return If adjusted for tuples and lists, ensuring structural
    equality in the 'then' and 'else' branches)."""
    if is_not_a(q,tuple,list):
        return If(p,q,r)
    return tuple([ If(p,a,b) for a,b in zip(list(q),list(r)) ])

def iff(a,b):
    """return bi-directional implication constraints"""
    return And(Implies(a,b),Implies(b,a))

def summation(values,xs):
    acc = 0
    cs = []
    for x in xs:
        new_acc = values('r')
        cs += [new_acc == x + acc]
        acc = new_acc
    return acc,cs

def pick_exactly_one(indicators):
    """return constraints choosing exactly one of several options"""
    some_value = Or(*indicators)
    K = len(indicators)
    unique_value = [ Not(And(indicators[j],indicators[i]))
                     for i in range(K)
                     for j in range(i+1,K) ]
    return [some_value] + unique_value

def permutation_indicators(K,values):
    """make a KxK bool matrix, one bool is true in each row and column"""
    indicators = [ values("b",K) for j in range(K) ]
    one_per_column = [ pick_exactly_one(indicators[r]) for r in range(K) ]
    one_per_row = [ pick_exactly_one([indicators[i][c] for i in range(K) ])
                    for c in range(K) ]
    constraints = list(chain.from_iterable(one_per_column)) + \
                  list(chain.from_iterable(one_per_row))
    return indicators,constraints

def apply_permutation(p, xs):
    """reorder xs according to permutation p"""
    return [ multiplexer(p[j],xs) for j in range(len(xs)) ]

def constrain_angle(dx,dy):
    """constrain dx,dy to be on the unit circle"""
    return [ dx*dx + dy*dy == 1 ]
