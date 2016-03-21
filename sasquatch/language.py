import math
from   .utilities   import is_not_a
from   .constraints import *
from   .values      import extract_real,extract_bool,extract_int

def is_primitive(rules):
    return is_not_a(rules,list)

class Language(object):
    """Sasquatch's meta-language for representation languages"""

    def __init__(self,values):
        self.rule_bank = {}
        self.values = values

    def rule(self,name, children, printer, evaluator, constraints):
        """add an arbitrary rule to the rule bank"""
        self.rule_bank[name] = self.rule_bank.get(name,[]) + \
                               [(children,printer,evaluator,constraints)]

    def indexed_rule(self,production, array_name, l, array_values):
        """add rules for indexing into an array (with mutable values?)"""
        def make_index(k):
            self.rule(production,[],
                 lambda m: "%s[%i]" % (array_name,k),
                 lambda i: array_values(i)[k],
                 [])
        for j in range(l):
            make_index(j)

    def enum_rule(self,production, options):
        """add rules for indexing into an enumeration"""
        def make_index(k):
            self.rule(production,[],
                 lambda m: str(options[k]),
                 lambda i: options[k],
                 [])
        for j in range(len(options)):
            make_index(j)

    def primitive_rule(self,name, callback):
        """add rules governing primitive values to the rule bank"""
        self.rule_bank[name] = callback

    def add_bools(self):
        def primitive_Boolean():
            """gives a tuple representing a bool"""
            thing = self.values("b")
            mdl = 1.0
            def eval_bool(i): return thing
            def print_bool(m): return extract_bool(m,thing)
            return eval_bool,mdl,print_bool,[]
        self.primitive_rule("BOOL",primitive_Boolean)

    def add_integers(self):
        def primitive_integer():
            """gives a tuple representing an integer"""
            thing = self.values("i")
            mdl = 10.0
            def eval_int(i): return thing
            def print_int(m): return extract_int(m,thing)
            return eval_int,mdl,print_int,[]
        self.primitive_rule("INT",primitive_integer)

    def add_reals(self):
        def primitive_real():
            """gives a tuple representing a real"""
            thing = self.values("r")
            mdl = 10.0
            def eval_real(i): return thing
            def print_real(m): return extract_real(m,thing)
            return eval_real,mdl,print_real,[]
        self.primitive_rule("REAL",primitive_real)

    def add_angles(self):
        def primitive_angle():
            """gives a tuple representing an angle"""
            x,y = (self.values("r"), self.values("r"))
            mdl = 10.0
            def eval_angle(i): return x,y
            def print_angle(m):
                return '(angle %i)' % int(math.atan2(extract_real(m,y),
                                                     extract_real(m,x)) *
                                          180.0/math.pi)
            return eval_angle,mdl,print_angle,constrain_angle(x,y)
        self.primitive_rule("ANGLE",primitive_angle)
