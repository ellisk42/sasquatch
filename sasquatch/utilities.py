"""utilities could appear in any project but happen to be useful here"""

import math

def yes(x):
    """is x the 'True' string?"""
    return str(x) == "True"

def logarithm(n):
    """base 2 logarithm"""
    return math.log(n)/math.log(2.0)

def distribution_mode(d):
    """give the modal item in some container"""
    k = {}
    for x in d:
        k[x] = k.get(x,0) + 1
    return max([ (y,x) for x,y in k.items() ])[1]

def is_a(obj,*args):
    """test whether obj is one of several kinds"""
    return any([isinstance(obj,kind) for kind in args])

def is_not_a(obj,*args):
    """test whether obj is not one of several kinds"""
    return not is_a(obj,args)
