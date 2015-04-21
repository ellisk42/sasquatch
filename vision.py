from solverUtilities import *
import math
import sys
import time

LD = 4 # latent distances
LA = 0 # latent angles
LS = 1 # latent shapes

'''
Weird thing: the solver chokes if I don't represent shape constants as real numbers.
'''

observations = []
for picture_file in sys.argv[1:]:
    if picture_file[0] != 'p':
        picture_file = "pictures/" + picture_file
    with open(picture_file,'r') as picture:
        picture = picture.readlines()
        shapes = eval('['+picture[0]+']')
        LS = max([LS] + [ shape[3]+1 for shape in shapes ])
        observations.append([(x,y,s+10.0*len(observations)) for [x,y,sz,s] in shapes ])
#observations = [ [(3.0*i,4.0*i,triangle) for i in [1,2,3]], #,(3,9,5)],
#                 [(1.0*i,1.0*i,rectangle) for i in [1,2,3]]] #,(7,10,2)] ]

dataMDL = len(observations)*(10.0*(LA+LD)+100.0*LS)
    
def move_turtle(x,y,tx,ty,d,ax,ay):
    constrain(d > 0)
    # compute new angle
    nx = real()
    ny = real()
    
    constrain(nx == tx*ax-ty*ay)
    constrain(ny == tx*ay+ty*ax)
    
    return x+d*nx,y+d*ny,nx,ny

rule('ORIENTATION', ['ANGLE'],
     lambda m, a: a,
     lambda i, a: a)
rule('ORIENTATION', [],
     lambda m: "0deg",
     lambda i: (1.0,0.0))
indexed_rule('ORIENTATION', 'a', LA,
             lambda (t,i): i['angles'])

indexed_rule('DISTANCE', 'l', LD,
             lambda (t,i): i['distances'])


rule('LOCATE', ['DISTANCE','DISTANCE'],
     lambda m, p, q: "(teleport %s %s)" % (p,q),
     lambda (t,i), p, q: (p,q,0.0,1.0))
rule('LOCATE', ['DISTANCE','ORIENTATION'],
     lambda m, d, o: "(move %s %s)" % (d,o),
     lambda ((x,y,dx,dy),i), d, (dxp,dyp): move_turtle(x,y,dx,dy,d,dxp,dyp))

rule('INITIALIZE',[],
     lambda m: "(teleport l[0] l[1])",
     lambda (t,i): (i['distances'][0],i['distances'][1],0.0,1.0))
rule('INITIAL-SHAPE',[],
     lambda m: "s[0]",
     lambda (t,i): i['shapes'][0])

def indexed_shape(j):
    rule('SHAPE',[],
         lambda m: "s[%i]" % j,
         lambda (t,i): i['shapes'][j])
for j in range(LS): indexed_shape(j)


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
    e = 0.01
    x,y,sh = shape
    xp,yp,sp = shapep
    if e > 0:
        return [x < xp + e, x > xp - e,
                y < yp + e, y > yp - e,
                sh == sp]
    return [x == xp,y == yp,sh == sp]
    

inputs = []
for n in range(len(observations)):
    ds = real_numbers(LD)
    ts = [ (real(), real()) for j in range(LA) ]
    for tx,ty in ts:
        constrain_angle(tx,ty)
    ss = real_numbers(LS)
    inputs.append(((0,0,1,0),{"distances": ds, "angles": ts, "shapes": ss}))

draw_picture,mdl,pr = program_generator(len(observations[0]))
mdl = summation([mdl,dataMDL])
for i in range(len(observations)):
    observation = observations[i]
    picture = draw_picture(inputs[i])
    permutation = permutation_indicators(len(picture))
    for r in range(len(picture)):
        for c in range(len(picture)):
            for constraint in check_shape(picture[r],observation[c]):
                constrain(Implies(permutation[r][c], constraint))


# modify printer so it also includes the latent dimensions
def full_printer(m):
    program = pr(m)+"\n"
    for n in range(len(observations)):
        program += "\nObservation %i:\n\t" % n
        for d in range(LD):
            program = program + ("l[%s] = %f; " % (str(d), extract_real(m,inputs[n][1]['distances'][d])))
        program += "\n\t"
        if LA > 0:
            for (dx,dy),index in zip(inputs[n][1]['angles'], range(LA)):
                a = int(math.atan2(extract_real(m,dy),extract_real(m,dx))/math.pi*180.0)
                program = program + ("a%i[%i] = %i; " % (n, index, a))
            program += "\n\t"
        for sh in range(LS):
            program = program + ("s[%s] = %f; " % (str(sh), extract_real(m,inputs[n][1]['shapes'][sh])))
        program = program + "\n"
    return program





compressionLoop(full_printer,mdl) #,[[boolean()]])
