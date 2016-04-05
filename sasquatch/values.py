"""ways to create Z3 values and extract information from them"""

from z3 import *

def valueMaker():
    """return a function generating uniquely-named symbols"""
    nextGenSym = [0]
    d = dict(i=Int,int=Int,b=Bool,bool=Bool,r=Real,real=Real)
    def new_symbol():
        """generate unique strings by using a counter"""
        nextGenSym[0] += 1
        return "gensym{0}".format(nextGenSym[0])
    def values(kind,N=None):
        """generate uniquely-named symbols"""
        if N:
            return [d[kind](new_symbol()) for n in range(N)]
        else:
            return d[kind](new_symbol())
    return values

def extract_bool(m,b):
    """converts a Z3 variable to a Boolean"""
    return None if (not m or m[b] == None) else str(m[b])

def extract_int(m,i):
    """converts a Z3 variable to an integer"""
    return None if (not m or m[i] == None) else m[i].as_long()

def extract_real(m,r):
    """converts a Z3 variable to a float"""
    if (not m or m[r] == None):
        return None
    else:
        return float(m[r].as_decimal(3).replace('?',''))
