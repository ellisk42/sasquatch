from solverUtilities import *
import math
import sys
import time
import re


translational_noise = 3
solver_timeout = 30

CONTAINS = True
BORDERS = False

NORMALSHAPE = True
SMALLSHAPE = False

LK = 0 # latent containments
LB = 0 # latent borders
LS = 0 # latent shapes

'''
Weird thing: the solver chokes if I don't represent shape constants as real numbers.
'''

class Shape():
    def __init__(self,x,y,name,scale,orientation):
        self.x = x
        self.y = y
        self.name = name
        self.scale = scale
        self.orientation = orientation
    def convert_to_tuple(self):
        return (self.x,self.y,self.name,self.scale,self.orientation)
    def __str__(self):
        return "%i@(%i,%i)x%f/%ideg" % (self.name, self.x, self.y, self.scale, self.orientation)
        

class Observation:
    def __init__(self,c,k,b):
        self.coordinates = c
        self.containment = k
        self.bordering = b


observations = []
test_observations = []
reading_test_data = False
for picture_file in sys.argv[1:]:
    if picture_file == 'test':
        reading_test_data = True
        continue
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
        if not reading_test_data:
            print "PICTURE:\n\t", '\t'.join([ str(s) for s in shapes ])
            print '\tBORDERS: %s' % str(borderings)
            print '\tCONTAINS: %s' % str(containment)
        composite = Observation([ss for ss in shapes ],
                                containment, borderings)
        if not reading_test_data:
            observations.append(composite)
        else:
            test_observations.append(composite)

# remove incorrectly parsed training data
picture_size = distribution_mode([ len(observation.coordinates) 
                                   for observation in observations ])
observations = [o for o in observations
                if len(o.coordinates) == picture_size]

LS = int(max([ shape.name for o in observations
               for shape in o.coordinates ]))
LK = max([ len(o.containment) for o in observations])
LB = max([ len(o.bordering) for o in observations])

# determine number of latent scaling variables
LZ = 0
if all([ any([ s.scale < 1.0 for s in o.coordinates ]) for o in observations ]):
    LZ = 1
# determine number of latent rotation variables
LR = 0
if all([ any([ s.orientation > 0.0 for s in o.coordinates ]) for o in observations ]):
    LR = 1


    
def move_turtle(x,y,tx,ty,d,ax,ay):
    constrain(d > 0)
    # compute new angle
    nx = real()
    ny = real()
    
    constrain(nx == tx*ax-ty*ay)
    constrain(ny == tx*ay+ty*ax)
    
    return x+d*nx,y+d*ny,nx,ny

def define_grammar(LP,LD,LA):
    if LD > 0:
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
             lambda i,s,z: (s[0],z*s[1],s[2]))
        rule('SHAPE-SIZE',[],
             lambda m: '',
             lambda (t,i): i['scale'])
        rule('INITIAL-SHAPE',['SHAPE-SIZE'],
             lambda m,z: '(draw s[0] :scale z)',
             lambda (t,i),z: (i['shapes'][0][0],i['shapes'][0][1]*z,i['shapes'][0][2]))
    if LR > 0:
        rule('SHAPE',['SHAPE-INDEX','SHAPE-SIZE','SHAPE-ORIENTATION'],
             lambda m,i,z,o: '(draw %s :scale z :orientation o)' % i,
             lambda i,s,z,o: (s[0],z*s[1],o+s[2]))
        rule('INITIAL-SHAPE',['SHAPE-SIZE','SHAPE-ORIENTATION'],
             lambda m,z,o: '(draw s[0] :scale z :orientation o)',
             lambda (t,i),z,o: (i['shapes'][0][0],i['shapes'][0][1]*z,i['shapes'][0][2]+o))
        rule('SHAPE-ORIENTATION',[],
             lambda m: '',
             lambda (t,i): i['orientation'])
        
    
    rule('DRAW-ACTION',['LOCATE','SHAPE'],
         lambda m,l,s: l + "\n" + s,
         lambda state, (x,y,dx,dy), (s,z,o): ((x,y,s,z,o),(x,y,dx,dy)))
    rule('INITIAL-DRAW',['INITIALIZE','INITIAL-SHAPE'],
         lambda m,l,s: l + "\n" + s,
         lambda state, (x,y,dx,dy), (s,z,o): ((x,y,s,z,o),(x,y,dx,dy)))
    
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
            shape.scale == shapep.scale,
            shape.orientation == shapep.orientation]

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
        
        
    

def make_new_input(LA,LD,LP):
    ps = [ (real(), real()) for j in range(LP) ]
    ds = real_numbers(LD)
    ts = [ (real(), real()) for j in range(LA) ]
    z = None if LZ == 0 else real()
    o = None if LR == 0 else real()
    for tx,ty in ts:
        constrain_angle(tx,ty)
    if LD > 0:
        ix,iy = real(), real()
        constrain_angle(iy,ix)
    else:
        ix,iy = None, None
    ss = real_numbers(LS)
    zs = real_numbers(LS)
    os = real_numbers(LS)
    return ((0,0,1,0),{"distances": ds, "angles": ts, "positions": ps, 
                       "shapes": zip(ss,zs,os), 
                       "scale": z, "orientation": o,
                       "initial-dx": ix, "initial-dy": iy})
        

solutions = []    
for LA,LD,LP in [(a,d,p) for a in [0,1] for d in [0,1,2] for p in range(1,picture_size+1) ]:
    # make sure that the latent dimensions make sense
    if LA > LD: continue
    if LP + LD > picture_size: continue
    # do you have a latent initial rotation?
    LI = LD > 0
    
    clear_solver()
    define_grammar(LP, LD, LA)
       
    draw_picture,mdl,pr = imperative_generator('DRAW-ACTION',picture_size,initial_production = 'INITIAL-DRAW')
    containment,containment_length,containment_printer = imperative_generator('TOPOLOGY-CONTAINS',LK)
    containment = containment(None,None)
    containment_data = summation([ If(t[0],0,logarithm(2)) for t in containment ])
    borders,borders_length,borders_printer = imperative_generator('TOPOLOGY-BORDERS',LB)
    borders = borders(None,None)
    borders_data = summation([ If(t[0],0,logarithm(2)) for t in borders ])
    dataMDL = len(observations)*(10.0*(LR+LZ+LA+LD+2*LP+LI)+100.0*LS+containment_data+borders_data)
    mdl = summation([mdl,dataMDL,containment_length,borders_length])
    
    # Push a frame to hold all of the training data
    push_solver()
    inputs = [ make_new_input(LA,LD,LP) for n in range(len(observations)) ]

    for i in range(len(observations)):
        picture = draw_picture(*inputs[i])
        check_picture(Observation(picture,containment,borders),observations[i])
    
    # If we have a solution so far, ensure that we beat it
    if len(solutions) > 0:
        bestLength = min(solutions)[0]
        constrain(mdl < bestLength)
    
    # modify printer so it also includes the latent dimensions
    def full_printer(m):
        program = pr(m)+containment_printer(m)+borders_printer(m)+"\n"
        for n in range(len(observations)):
            program += "\nObservation %i:\n\t" % n
            for d in range(LP):
                program = program + ("r[%s] = (%f,%f); " % (str(d),
                                                            extract_real(m,inputs[n][1]['positions'][d][0]),
                                                            extract_real(m,inputs[n][1]['positions'][d][1])))
                program += "\n\t"
            if LD > 0:
                a = int(math.atan2(extract_real(m,inputs[n][1]['initial-dy']),
                                   extract_real(m,inputs[n][1]['initial-dx']))/math.pi*180.0)
                program += "initial-orientation = %f;\n\t" % a
                for d in range(LD):
                    program = program + ("l[%s] = %f; " % (str(d), 
                                                           extract_real(m,inputs[n][1]['distances'][d])))
                program += "\n\t"
            if LA > 0:
                for (dx,dy),index in zip(inputs[n][1]['angles'], range(LA)):
                    a = int(math.atan2(extract_real(m,dy),extract_real(m,dx))/math.pi*180.0)
                    program = program + ("a[%i] = %i; " % (index, a))
                program += "\n\t"
            for sh in range(LS):
                program = program + ("s[%i] = %f; " % (sh, extract_real(m,inputs[n][1]['shapes'][sh][0])))
                program = program + ("s_scale[%i] = %f; " % (sh, extract_real(m,inputs[n][1]['shapes'][sh][1])))
                program = program + ("s_orientation[%i] = %f; " % (sh, extract_real(m,inputs[n][1]['shapes'][sh][2])))
            if LZ > 0:
                program += "\n\tz = %f" % extract_real(m,inputs[n][1]['scale'])
            if LR > 0:
                program += "\n\to = %f" % extract_real(m,inputs[n][1]['orientation'])
            program = program + "\n"
        return program
    print "Trying LA, LD, LP = %i, %i, %i" % (LA,LD,LP)
    p,m = compressionLoop(full_printer,mdl,timeout = solver_timeout)
    if m == None:
        print "No solution for LA, LD, LP = %i, %i, %i" % (LA,LD,LP)
    else:
        print "Got solution for LA, LD, LP = %i, %i, %i" % (LA,LD,LP)
        kd = extract_real(get_recent_model(),containment_data) if LK > 0 else 0
        bd = extract_real(get_recent_model(),borders_data) if LB > 0 else 0
        solutions.append((m,p,LA,LD,LP,get_solver(),draw_picture,containment,kd,borders,bd))



# Failure to synthesize any program
if len(solutions) == 0:
    print "Failure to synthesize"
    if test_observations:
        print "Test data log likelihoods:"
        for t in test_observations:
            print "-infinity"
    else:
        print "infinity"
    sys.exit(0)

(m,p,LA,LD,LP,solver,gen,k,kd,b,bd) = min(solutions)
LI = LD > 0 # initial rotation
print "="*40
print "Best solution: %f bits (D,A,P = %i,%i,%i)" % (m,LD,LA,LP)
print "="*40

print p

set_solver(solver)
if test_observations:
    print "Test data log likelihoods:"
    for test in test_observations:
        if (len(test.coordinates) != picture_size
            or max([ shape.name for shape in test.coordinates ]) > LS
            or len(test.containment) > LK
            or len(test.bordering) > LB):
            print "-infinity"
            continue
        push_solver()
        inputs = make_new_input(LA,LD,LP)
        outputs = gen(*inputs)
        check_picture(Observation(outputs,
                                  k,
                                  b),
                      test)
        if 'sat' == str(solver.check()):
            print "-%f" % (10.0*(LR+LZ+LA+LD+2*LP+LI)+100.0*LS+kd+bd)
        else:
            print "-infinity"
        pop_solver()
else:
    print m
        
