from solverUtilities import *
import math
import sys
import time
import re


MDL_SHAPE = 1000
MDL_REAL = 10

translational_noise = 3
JITTER = 7
solver_timeout = 30


LK = 0 # latent containments
LB = 0 # latent borders
LS = 0 # latent shapes

VERBOSE = True
def vision_verbosity(v):
    global VERBOSE
    VERBOSE = v

class Shape():
    def __init__(self,x,y,name,scale,orientation = 0.0):
        self.x = x
        self.y = y
        self.name = name
        # convert the logarithmic scale
        if isinstance(scale,float):
            if scale > 0.0:
                scale = math.log(scale)
            if scale > math.log(0.98):
                scale = 0.0
        self.scale = scale
    def convert_to_tuple(self):
        return (self.x,self.y,self.name,self.scale)
    def __str__(self):
        if self.scale == 0.0:
            return "%i@(%i,%i)" % (self.name,self.x,self.y)
        return "%i@(%i,%i)x%f" % (self.name, self.x, self.y, math.exp(self.scale))
        

class Observation:
    def __init__(self,c,k,b):
        self.coordinates = c
        self.containment = k
        self.bordering = b
    def show(self):
        print '\t'.join([ str(s) for s in self.coordinates ])
        if len(self.bordering) > 0: print '\tBORDERS: %s' % str(self.bordering)
        if len(self.containment) > 0: print '\tCONTAINS: %s' % str(self.containment)

def load_picture(picture_file):
    if picture_file[0] != 'p':
        picture_file = "pictures/" + picture_file
    with open(picture_file,'r') as picture:
        picture = picture.readlines()
        shapes = eval('['+picture[0]+']')
        # parse qualitative information
        containment = []
        borderings = []
        for l in picture[1:]:
            k = re.match(r'contains\(([0-9]+), ?([0-9]+)\)',l)
            b = re.match(r'borders\(([0-9]+), ?([0-9]+)\)',l)
            if k:
                containment.append((int(k.group(1)),int(k.group(2))))
            if b:
                borderings.append((int(b.group(1)),int(b.group(2))))
        composite = Observation([ss for ss in shapes ],
                                containment, borderings)
        return composite

def load_pictures(files, show = False):
    o = [load_picture(f) for f in files ]
    if show:
        for j in range(len(files)):
            print "PICTURE:",files[j],
            o[j].show()
    return o

def parse_arguments():
    global VERBOSE
    if sys.argv[1] == '--quiet':
        VERBOSE = False
        sys.argv.remove('--quiet')
    observations = []
    test_observations = []
    reading_test_data = False
    for p in sys.argv[1:]:
        if p == 'test':
            reading_test_data = True
            continue
        if reading_test_data:
            test_observations.append(p)
        else:
            observations.append(p)
    return observations,test_observations



def set_degrees_of_freedom(observations):
    global picture_size,LS,LK,LB
    # remove incorrectly parsed training data
    picture_size = distribution_mode([ len(observation.coordinates) 
                                       for observation in observations ])
    observations = [o for o in observations
                    if len(o.coordinates) == picture_size]

    LS = int(max([ shape.name for o in observations
                   for shape in o.coordinates ]))
    LK = max([ len(o.containment) for o in observations])
    LB = max([ len(o.bordering) for o in observations])

    return observations
    
def move_turtle(x,y,tx,ty,d,ax,ay):
    constrain(d > 0)
    # compute new angle
    nx = real()
    ny = real()
    
    constrain(nx == tx*ax-ty*ay)
    constrain(ny == tx*ay+ty*ax)
    
    return x+d*nx,y+d*ny,nx,ny

def define_grammar(LZ,LP,LD,LA):
    if LD > 0:
        if LD == 1 and LA == 0:
            rule('MOVE',[],
                 lambda m: "(move l[0] 0deg)",
                 lambda ((x,y,dx,dy),i): (x+dx,y+dy,dx,dy))
            rule('MOVE',['TURN'],
                 lambda m,t: t,
                 lambda m,t: t)
            rule('TURN',[],
                 lambda m: '(move l[0] 90deg)',
                 lambda ((x,y,dx,dy),i): (x-dy,y+dx,-dy,dx))
            rule('TURN',[],
                 lambda m: '(move l[0] -90deg)',
                 lambda ((x,y,dx,dy),i): (x+dy,y-dx,dy,-dx))
            rule('LOCATE',['MOVE'],
                 lambda m,l: l,
                 lambda i,l: l)
        else:
            rule('ORIENTATION', [],
                 lambda m: "0deg",
                 lambda i: (1.0,0.0))

            rule('TURN', [],
                 lambda m: "90deg",
                 lambda i: (0.0,1.0))
            rule('TURN', [],
                 lambda m: "-90deg",
                 lambda i: (0.0,-1.0))
            rule('ORIENTATION',['TURN'],
                 lambda m,t: t,
                 lambda i,t: t)
            indexed_rule('ORIENTATION', 'a', LA,
                         lambda (t,i): i['angles'])
            rule('LOCATE', ['DISTANCE','ORIENTATION'],
                 lambda m, d, o: "(move %s %s)" % (d,o),
                 lambda ((x,y,dx,dy),i), d, (dxp,dyp): move_turtle(x,y,dx,dy,d,dxp,dyp))
            indexed_rule('DISTANCE', 'l', LD,
                         lambda (t,i): i['distances'])
    
    indexed_rule('POSITION', 'r', LP,
                 lambda (t,i): i['positions'])
    rule('LOCATE', ['POSITION'],
         lambda m, p: "(teleport %s)" % p,
         lambda (t,i), (p, q): (p,q,
                                i['initial-dx'] if i['initial-dx'] else 1.0,
                                i['initial-dy'] if i['initial-dy'] else 0.0))
    rule('LOCATE', ['FLIP'],
         lambda m, f: f,
         lambda (t,i), f: f)
    rule('FLIP',[],
         lambda m: '(flip-Y)',
         lambda ((x,y,dx,dy),i): (x,128-y,dx,dy))
    rule('FLIP',[],
         lambda m: '(flip-X)',
         lambda ((x,y,dx,dy),i): (128-x,y,dx,dy))
    
    rule('LOCATE',['JITTER'],
         lambda m,j: j,
         lambda ((x,y,dx,dy),i), (jx,jy): (x+jx,y+jy,dx,dy))
    rule('JITTER',[],
         lambda m: '(jitter)',
         lambda (s,i): (i['jitter-x'],i['jitter-y']))

    
    rule('INITIALIZE',[],
         lambda m: "(teleport r[0])",
         lambda (t,i): (i['positions'][0][0],i['positions'][0][1],
                        i['initial-dx'] if i['initial-dx'] else 1.0,
                        i['initial-dy'] if i['initial-dy'] else 0.0))
    rule('INITIAL-SHAPE',[],
         lambda m: '(draw s[0])',
         lambda (t,i): i['shapes'][0])
    
    indexed_rule('SHAPE-INDEX', 's', LS,
                 lambda (t,i): i['shapes'])
    
    rule('SHAPE',['SHAPE-INDEX'],
         lambda m,i: '(draw %s)' % i,
         lambda i,s: s)
    if LZ > 0:
        rule('SHAPE',['SHAPE-INDEX','SHAPE-SIZE'],
             lambda m,i,z: '(draw %s :scale z)' % i,
             lambda i,s,z: (s[0],z+s[1]))
        rule('SHAPE-SIZE',[],
             lambda m: '',
             lambda (t,i): i['scale'])
        rule('INITIAL-SHAPE',['SHAPE-SIZE'],
             lambda m,z: '(draw s[0] :scale z)',
             lambda (t,i),z: (i['shapes'][0][0],i['shapes'][0][1]+z))
    
    rule('DRAW-ACTION',['LOCATE','SHAPE'],
         lambda m,l,s: l + "\n" + s,
         lambda state, (x,y,dx,dy), (s,z): ((x,y,s,z),(x,y,dx,dy)))
    rule('INITIAL-DRAW',['INITIALIZE','INITIAL-SHAPE'],
         lambda m,l,s: l + "\n" + s,
         lambda state, (x,y,dx,dy), (s,z): ((x,y,s,z),(x,y,dx,dy)))
    
    rule('TOPOLOGY-OPTION',[],
         lambda m: "",
         lambda n: True)
    rule('TOPOLOGY-OPTION',[],
         lambda m: "-option",
         lambda n: False)
    def define_adjacency(i,j):
        if i == j: return
        il = [False]*picture_size
        il[i] = True
        jl = [False]*picture_size
        jl[j] = True
        il,jl = tuple(il),tuple(jl)
        rule('TOPOLOGY-CONTAINS',['TOPOLOGY-OPTION'],
             lambda m, o: "(assert (contains%s %i %i))" % (o,i,j),
             lambda n, o: ((o,il,jl),True))
        if i < j:
            rule('TOPOLOGY-BORDERS',['TOPOLOGY-OPTION'],
                 lambda m,o: "(assert (borders%s %i %i))" % (o,i,j),
                 lambda n,o: ((o,il,jl),True))
    for i in range(picture_size):
        for j in range(picture_size):
            define_adjacency(i,j)



# returns an expression that asserts that the shapes are equal
def check_shape(shape, shapep):
    e = translational_noise
    return [shape.x <= shapep.x + e, shape.x >= shapep.x - e,
            shape.y <= shapep.y + e, shape.y >= shapep.y - e,
            shape.name == shapep.name,
            shape.scale == shapep.scale]

# adds a constraint saying that the picture has to equal some permutation of the observation
def check_picture(picture,observation):
    permutation = permutation_indicators(len(picture.coordinates))
    permuted_picture = apply_permutation(permutation, picture.coordinates)
    permuted_picture = [ Shape(*p) for p in permuted_picture ]
    for shape1, shape2 in zip(permuted_picture, observation.coordinates):
        constrain(check_shape(shape1, shape2))
    # permuted topology
    picture_containment = [(mandatory,apply_permutation(permutation,l),apply_permutation(permutation,r))
                           for (mandatory,l,r) in picture.containment ]
    picture_bordering = [(mandatory,apply_permutation(permutation,l),apply_permutation(permutation,r))
                           for (mandatory,l,r) in picture.bordering ]
    for (mandatory,l,r) in picture_containment:
        constrain(Implies(mandatory,
                          Or(*[And(l[i],r[j]) for i,j in observation.containment ])))
    for i,j in observation.containment:
        constrain(Or(*[And(l[i],r[j]) for (mandatory,l,r) in picture_containment ]))
    for (mandatory,l,r) in picture_bordering:
        constrain(Implies(mandatory,
                          Or(*([And(l[i],r[j]) for i,j in observation.bordering ] + 
                               [And(l[j],r[i]) for i,j in observation.bordering ]))))
    for i,j in observation.bordering:
        constrain(Or(*([And(l[i],r[j]) for (mandatory,l,r) in picture_bordering ] + 
                       [And(l[j],r[i]) for (mandatory,l,r) in picture_bordering ])))
        
        
    

def make_new_input(LZ,LA,LD,LP):
    ss = real_numbers(LS)
    zs = real_numbers(LS)
    jx = real()
    jy = real()
    constrain(jx < JITTER)
    constrain(jx > -JITTER)
    constrain(jy < JITTER)
    constrain(jy > -JITTER)
    ps = [ (real(), real()) for j in range(LP) ]
    z = None if LZ == 0 else real()
    if LZ > 0:
        constrain(z < 0.0)
    
    is_linear = LA == 0 and LD == 1

    ds = None if is_linear else real_numbers(LD)
    ts = [ (real(), real()) for j in range(LA) ]
    for tx,ty in ts:
        constrain_angle(tx,ty)
    if LD > 0:
        ix,iy = real(), real()
        if not is_linear:
            constrain_angle(iy,ix)
    else:
        ix,iy = None, None
    return ((0,0,1,0),{"distances": ds, "angles": ts, "positions": ps, 
                       "shapes": zip(ss,zs), 
                       "scale": z,
                       "jitter-x": jx, "jitter-y": jy,
                       "initial-dx": ix, "initial-dy": iy})
        

def grid_search(observations):
    observations = set_degrees_of_freedom(observations)
    solutions = []    
    LZ = 0
    for LZ,LA,LD,LP in [(z,a,d,p) for z in [0,1]  for a in [0,1] for d in [0,1,2] for p in range(1,picture_size+1) ]:
        # make sure that the latent dimensions make sense
        if LA > LD: continue
        if LP + LD > picture_size: continue
        # do you have a latent initial rotation?
        LI = LD > 0

        clear_solver()
        define_grammar(LZ, LP, LD, LA)

        draw_picture,mdl,pr = imperative_generator('DRAW-ACTION',picture_size,initial_production = 'INITIAL-DRAW')
        containment,containment_length,containment_printer = imperative_generator('TOPOLOGY-CONTAINS',LK)
        containment = containment(None,None)
        containment_data = summation([ If(t[0],0,logarithm(2)) for t in containment ])
        borders,borders_length,borders_printer = imperative_generator('TOPOLOGY-BORDERS',LB)
        borders = borders(None,None)
        borders_data = summation([ If(t[0],0,logarithm(2)) for t in borders ])
        dataMDL = len(observations)*(MDL_REAL*(LZ+LA+LD+2*LP+LI)+MDL_SHAPE*LS+containment_data+borders_data)
        mdl = summation([mdl,dataMDL,containment_length,borders_length])

        # Push a frame to hold all of the training data
        push_solver()
        inputs = [ make_new_input(LZ,LA,LD,LP) for n in range(len(observations)) ]

        for i in range(len(observations)):
            picture = draw_picture(*inputs[i])
            check_picture(Observation(picture,containment,borders),observations[i])

        # If we have a solution so far, ensure that we beat it
        if len(solutions) > 0:
            bestLength = min(solutions)[0]
            constrain(mdl < bestLength)

        # modify printer so it also includes the latent dimensions
        def full_printer(m):
            is_linear = inputs[0][1]['distances'] == None
            program = pr(m)+containment_printer(m)+borders_printer(m)+"\n"
            for n in range(len(observations)):
                program += "\nObservation %i:\n\t" % n
                for d in range(LP):
                    program = program + ("r[%s] = (%f,%f); " % (str(d),
                                                                extract_real(m,inputs[n][1]['positions'][d][0]),
                                                                extract_real(m,inputs[n][1]['positions'][d][1])))
                    program += "\n\t"
                if LD > 0:
                    dy = extract_real(m,inputs[n][1]['initial-dy'])
                    dx = extract_real(m,inputs[n][1]['initial-dx'])
                    a = int(math.atan2(dy,dx)/math.pi*180.0)
                    program += "initial-orientation = %f;\n\t" % a
                    if not is_linear:
                        for d in range(LD):
                            program = program + ("l[%s] = %f; " % (str(d), 
                                                                   extract_real(m,inputs[n][1]['distances'][d])))
                    else:
                        program += "l[0] = %f" % (math.sqrt(dx*dx+dy*dy))
                    program += "\n\t"
                if LA > 0:
                    for (dx,dy),index in zip(inputs[n][1]['angles'], range(LA)):
                        a = int(math.atan2(extract_real(m,dy),extract_real(m,dx))/math.pi*180.0)
                        program = program + ("a[%i] = %i; " % (index, a))
                    program += "\n\t"
                for sh in range(LS):
                    program = program + ("s[%i] = %f; " % (sh, extract_real(m,inputs[n][1]['shapes'][sh][0])))
                    program = program + ("s_scale[%i] = %f; " % (sh, extract_real(m,inputs[n][1]['shapes'][sh][1])))
                if LZ > 0:
                    program += "\n\tz = %f" % math.exp(extract_real(m,inputs[n][1]['scale']))
                program = program + "\n"
            return program

        if VERBOSE: print "Trying LZ, LA, LD, LP = %i, %i, %i, %i" % (LZ,LA,LD,LP)
        p,m = compressionLoop(full_printer,mdl,timeout = solver_timeout,verbose = VERBOSE)
        if m == None:
            if VERBOSE: print "No solution for LZ, LA, LD, LP = %i, %i, %i, %i" % (LZ,LA,LD,LP)
        else:
            if VERBOSE: print "Got solution for LZ, LA, LD, LP = %i, %i, %i, %i" % (LZ,LA,LD,LP)
            kd = extract_real(get_recent_model(),containment_data) if LK > 0 else 0
            bd = extract_real(get_recent_model(),borders_data) if LB > 0 else 0
            solutions.append((m,p,LZ,LA,LD,LP,get_solver(),draw_picture,containment,kd,borders,bd))
    return solutions


def compute_picture_likelihoods(observations,test_observations):
    observations = load_pictures(observations,show = VERBOSE)
    test_observations = load_pictures(test_observations)
    solutions = grid_search(observations)
    
    # Failure to synthesize any program
    if len(solutions) == 0:
        return float('-inf'), [float('-inf')]*len(test_observations)
        
    (m,p,LZ,LA,LD,LP,solver,gen,k,kd,b,bd) = min(solutions)
    LI = LD > 0 # initial rotation
    marginal = -m
    
    set_solver(solver)
    test_likelihoods = []
    for test in test_observations:
        if (len(test.coordinates) != picture_size
            or max([ shape.name for shape in test.coordinates ]) > LS
            or len(test.containment) > LK
            or len(test.bordering) > LB):
            test_likelihoods.append(float('-inf'))
            continue
        push_solver()
        inputs = make_new_input(LZ,LA,LD,LP)
        outputs = gen(*inputs)
        check_picture(Observation(outputs,
                                  k,
                                  b),
                      test)
        if 'sat' == str(solver.check()):
            test_likelihoods.append(-(MDL_REAL*(LZ+LA+LD+2*LP+LI)+MDL_SHAPE*LS+kd+bd))
        else:
            test_likelihoods.append(float('-inf'))
        pop_solver()
    return marginal,test_likelihoods
        
if __name__ == '__main__':
    observations,test_observations = parse_arguments()
    print compute_picture_likelihoods(observations,test_observations)
