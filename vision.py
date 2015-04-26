from solverUtilities import *
import math
import sys
import time

translational_noise = 7

#LD = 4 # latent distances
#LA = 0 # latent angles
LS = 1 # latent shapes

'''
Weird thing: the solver chokes if I don't represent shape constants as real numbers.
'''


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
        print "PICTURE: ", shapes
        if not reading_test_data:
            LS = max([LS] + [ shape[3]+1 for shape in shapes ])
            observations.append([(x,y,s+10.0*len(observations)) for [x,y,sz,s] in shapes ])
        else:
            test_observations.append([(x,y,s)
                                      for [x,y,sz,s] in shapes ])


    
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
#        rule('ORIENTATION', ['ANGLE'],
#             lambda m, a: a,
#             lambda i, a: a)
        rule('ORIENTATION', [],
             lambda m: "0deg",
             lambda i: (1.0,0.0))
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
         lambda (t,i), (p, q): (p,q,1.0,0.0))
    
    
    rule('INITIALIZE',[],
         lambda m: "(teleport r[0])",
         lambda (t,i): (i['positions'][0][0],i['positions'][0][1],1.0,0.0))
    rule('INITIAL-SHAPE',[],
         lambda m: "s[0]",
         lambda (t,i): i['shapes'][0])
    
    indexed_rule('SHAPE', 's', LS,
                 lambda (t,i): i['shapes'])



def program_generator(d):
    if d < len(observations[0]):
        l, m1, pl = generator(5,'LOCATE')
        sh, m2, ps = generator(5,'SHAPE')
    else:
        l, m1, pl = generator(5,'INITIALIZE')
        sh, m2, ps = generator(5,'INITIAL-SHAPE')
    
    rest_of_picture = lambda (t,i): []
    restMDL, restPrint = 0, (lambda m: "")
    if d > 1:
        rest_of_picture, restMDL, restPrint = program_generator(d-1)
    def pr(m):
        return pl(m) + "\n(draw " + ps(m) + ")\n" + restPrint(m)
    def ev((t,i)):
        tp = l((t,i))
        sp = sh((t,i))
        return [(tp[0],tp[1],sp)] + rest_of_picture((tp,i))
    
    mdl = real()
    constrain(mdl == m1+m2+restMDL)
    return ev, mdl, pr

# returns an expression that asserts that the shapes are equal
def check_shape(shape, shapep):
    e = translational_noise
    x,y,sh = shape
    xp,yp,sp = shapep
    if e > 0:
        return [x < xp + e, x > xp - e,
                y < yp + e, y > yp - e,
                sh == sp]
    return [x == xp,y == yp,sh == sp]

# adds a constraint saying that the picture has to equal some permutation of the observation
def check_picture(picture,observation):
    permutation = permutation_indicators(len(picture))
    permuted_picture = apply_permutation(permutation, picture)
    for shape1, shape2 in zip(permuted_picture, observation):
        constrain(check_shape(shape1, shape2))

def make_new_input(LA,LD,LP):
    ps = [ (real(), real()) for j in range(LP) ]
    ds = real_numbers(LD)
    ts = [ (real(), real()) for j in range(LA) ]
    for tx,ty in ts:
        constrain_angle(tx,ty)
    ss = real_numbers(LS)
    return ((0,0,1,0),{"distances": ds, "angles": ts, "shapes": ss, "positions": ps})
        

solutions = []    
for LA,LD,LP in [(a,d,p) for a in [0,1] for d in [0,1,2] for p in range(1,len(observations[0])+1) ]:
    # make sure that the latent dimensions make sense
    if LA > 0 and LD == 0: continue
    if LP + LD > len(observations[0]): continue
    
    clear_solver()
    define_grammar(LP, LD, LA)
       
    draw_picture,mdl,pr = program_generator(len(observations[0]))
    dataMDL = len(observations)*(10.0*(LA+LD+2*LP)+100.0*LS)
    mdl = summation([mdl,dataMDL])
    
    # Push a frame to hold all of the training data
    push_solver()
    inputs = [ make_new_input(LA,LD,LP) for n in range(len(observations)) ]

    for i in range(len(observations)):
        observation = observations[i]
        picture = draw_picture(inputs[i])
        check_picture(picture,observation)
    
    # If we have a solution so far, ensure that we beat it
    if len(solutions) > 0:
        bestLength = min(solutions)[0]
        constrain(mdl < bestLength)
    
    # modify printer so it also includes the latent dimensions
    def full_printer(m):
        program = pr(m)+"\n"
        for n in range(len(observations)):
            program += "\nObservation %i:\n\t" % n
            for d in range(LP):
                program = program + ("r[%s] = (%f,%f); " % (str(d),
                                                            extract_real(m,inputs[n][1]['positions'][d][0]),
                                                            extract_real(m,inputs[n][1]['positions'][d][1])))
                program += "\n\t"
            if LD > 0:
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
                program = program + ("s[%s] = %f; " % (str(sh), extract_real(m,inputs[n][1]['shapes'][sh])))
            program = program + "\n"
        return program
    p,m = compressionLoop(full_printer,mdl)
    if m == None:
        print "No solution for LA, LD, LP = %i, %i, %i" % (LA,LD,LP)
    else:
        print "Got solution for LA, LD, LP = %i, %i, %i" % (LA,LD,LP)
        solutions.append((m,p,LA,LD,LP,get_solver(),draw_picture))



(m,p,LA,LD,LP,solver,gen) = min(solutions)
print "="*40
print "Best solution: %f bits (D,A,P = %i,%i,%i)" % (m,LD,LA,LP)
print "="*40

print p

set_solver(solver)
if test_observations:
    print "Test data log likelihoods:"
    for test in test_observations:
        if len(test) != len(observations[0]) or max([ shape[2]+1 for shape in test ]) > LS:
            print "-infinity"
            continue
        push_solver()
        inputs = make_new_input(LA,LD,LP)
        outputs = gen(inputs)
        check_picture(outputs,test)
        if 'sat' == str(solver.check()):
            print "-%f" % (10.0*(LA+LD+2*LP)+100.0*LS)
        else:
            print "-infinity"
        pop_solver()
        
