import time
from   .generator import Generator
from   .values    import extract_real
from   z3         import *

class Sasquatch(object):
    """Sasquatch: the representation-learning monster"""

    def __init__(self,values,name,verbose=False,timeout=None,structured=True):
        """create a sasquatch"""
        self.name = name
        self.slv = Solver()
        self.values = values
        self.gen = None
        self.recent_model = None
        self.recent_total_time = None
        self.verbose = verbose
        self.structured = structured
        self.timeout = timeout
        self.accuracy_constraints = []
        self.efficiency_constraints = []
        self.structure_constraints = []

    def clear(self):
        """reset the solver, generator, and memory"""
        self.slv = Solver()
        self.gen = Generator(values)
        self.recent_model = None
        self.recent_total_time = None

    def make_search_space(self,language,prod,maxDepth=1):
        self.gen = Generator(self.values,language)
        return self.gen.generate(prod,maxDepth)

    def constrain(self,constraint):
        """add a generic constraint to the solver"""
        self.slv.add(*constraint)

    def set_timeout(self):
        """set the solver timeout, if any"""
        if self.timeout:
            self.slv.set("timeout",self.timeout*1000)

    def write_solver(self):
        """write out the current solver to disk"""
        with open(self.name + '.smt2','w') as f:
            f.write(self.slv.to_smt2())

    def check_solver(self,pr,mdl):
        self.slv.push()
        self.constrain(self.gen.constraints)
        self.constrain(self.accuracy_constraints)
        self.constrain(self.efficiency_constraints)
        if self.structured:
            self.constrain(self.structure_constraints)
        start_time = time.time()
        if self.verbose:
            print "Checking model using Z3..."
        check = self.slv.check()
        dt = (time.time() - start_time)
        if str(check) == 'sat':
            m = self.slv.model()
            self.recent_model = m
        else:
            m = None
        new_best = extract_real(m,mdl)
        if self.verbose and m:
            print "Found model in {0:.2f}s".format(dt)
            print pr(m)
            print "MDL = {0:.2f}".format(new_best)
        if self.verbose and not m:
            print "Proved unsatisfiable in {0:.2f}s".format(dt)
        self.slv.pop()
        return dt,m,new_best

    def compress(self,pr,mdl):
        """iteratively search for shorter descriptions of the training data"""
        self.write_solver()
        self.set_timeout()
        solver_time = 0
        best_m = None
        while True: # iteratively improve until no lower mdl model exists
            dt,new_m,new_mdl = self.check_solver(pr,mdl)
            solver_time += dt
            if new_m:
                self.constrain([mdl < new_mdl])
                best_m = new_m
            else:
                break
            if self.verbose:
                print "Trying for a better one..."
        self.recent_total_time = solver_time
        if self.verbose:
            print "Total solver time: {0:.2f}s".format(solver_time)
        if not best_m: # report failure
            return 'FAIL',None
        if self.structured:
            # these constraints restrict future uses of this Sasquatch
            # (e.g. new test cases) to the same program structure
            self.structure_constraints = [ v == best_m[v]
                                           for v in self.gen.structural_variables
                                           if best_m[v] != None ]
        return pr(best_m), extract_real(best_m,mdl)
