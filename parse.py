import os
import time
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import sys
import re

# remove anything smaller than this that is in contact with anything bigger than this
tiny_threshold = 50

# makes the mask bigger by one pixel around the borders
def engorge(mask):
    engorged = mask
    engorged = np.logical_or(engorged,np.roll(mask,1,0))
    engorged = np.logical_or(engorged,np.roll(mask,-1,0))
    engorged = np.logical_or(engorged,np.roll(mask,1,1))
    engorged = np.logical_or(engorged,np.roll(mask,-1,1))
    return engorged


class Shape():
    def __init__(self,mask):
        self.initialize(mask)
    def initialize(self,mask):
        self.mask = mask
        self.negative = np.logical_not(mask)
        self.mass = np.sum(mask)
        self.contains = []
        self.borders = []
        self.merge_borders = []
        self.name = None
        
        self.outline = engorge(engorge(self.mask))
        self.merge_outline = engorge(self.outline)
        
        # compute center of mass
        w = mask.shape[0]
        h = mask.shape[1]
        xv, yv = np.meshgrid(np.arange(0,w),np.arange(0,h))
        self.x = int(np.sum(self.mask*xv)/self.mass)
        self.y = int(np.sum(self.mask*yv)/self.mass)
        
        self.centered = np.roll(self.mask,w/2-self.x,1)
        self.centered = np.roll(self.centered,h/2-self.y,0)
        
    def same_shape(self,other):
        d = np.logical_xor(self.centered,other.centered)
        return np.sum(d) < 5
    def contains_other(self,other):
        return np.sum(np.logical_and(self.negative,other.mask)) == 0
    def touches(self,other):
        return np.sum(np.logical_and(self.outline,other.outline)) > 0
    def merge_touches(self,other):
        return np.sum(np.logical_and(self.merge_outline,other.merge_outline)) > 0
    def merge_with(self,other):
        new_mask = np.logical_or(self.mask,other.mask)
        self.initialize(new_mask)

# resets all of the qualitative information
def compute_qualitative(shapes):
    for s in shapes:
        s.contains = []
        s.borders = []
        s.merge_borders = []
    for s in shapes:
        for z in shapes:
            if s == z: continue
            if s.contains_other(z):
                s.contains.append(z)
    for s in shapes:
        for z in shapes:
            if s == z or s in z.contains or z in s.contains: continue
            if s.touches(z):
                s.borders.append(z)
            if s.merge_touches(z):
                s.merge_borders.append(z)



def fill(a,x,y,o,n):
    w = a.shape[0]
    h = a.shape[1]
    if x < w and y < h and x > -1 and y > -1 and a[x,y] == o:
        a[x,y] = n
        fill(a,x-1,y,o,n)
        fill(a,x+1,y,o,n)
        fill(a,x,y-1,o,n)
        fill(a,x,y+1,o,n)
        

def analyze(filename):
    i = scipy.misc.imread(filename,1)
    i = i.astype(np.int32) # [ [ int(x) for x in r ] for r in i]
    w = i.shape[0]
    h = i.shape[1]
    ns = 1 # next shape
    for x in xrange(0,w):
        for y in xrange(0,h):
            if i[x,y] == 255:
                fill(i,x,y,255,ns)
                ns = ns+1
    ns = ns-1
    
    # isolate connected regions
    shapes = np.zeros((ns,w,h),dtype = np.int32)
    for s in xrange(0,ns):
        shapes[s,:,:] = np.vectorize(lambda x: 255 if x == 0 else x)(i)
        shapes[s,:,:] = np.vectorize(lambda x: 255 if x != (s+1) else 0)(shapes[s,:,:])
        
    # remove background
    ns = ns-1
    shapes = shapes[1:,:,:]
    
   # fill in occluded shapes
    for s in xrange(0,ns):
        fill(shapes[s,:,:],0,0,255,42)
        shapes[s,:,:] = np.vectorize(lambda x: 255 if x == 42 else 0)(shapes[s,:,:])
    
    # pack everything up into objects
    shapes = [ Shape(shapes[s,:,:] < 255) for s in range(ns) ]

    # merger artifacts with what they came from
    changed = True
    while changed:
        changed = False
        compute_qualitative(shapes)
        to_remove = None
        for s,z in [(s,z) for s in shapes for z in shapes ]:
            if s == z: continue
            if z.mass < tiny_threshold and s.mass >= z.mass and z in s.merge_borders:
                s.merge_with(z)
                to_remove = z
                break
        if to_remove:
            changed = True
            shapes.remove(to_remove)
    ns = len(shapes)
    # labeled the shapes
    labeled = []
    next_label = 1
    for s in shapes:
        for l in labeled:
            if l.same_shape(s):
                s.name = l.name
                break
        if not s.name:
            s.name = next_label
            next_label += 1
        labeled.append(s)
    # build output string
    os = []
    for s in shapes:
        os.append("[" + str(s.x) + ", " + str(s.y) + ", " + str(s.mass) + ", " + str(s.name)+"]")
    os = ','.join(os)
    os = os + "\n"
    for s in xrange(0,ns):
        for sp in xrange(0,ns):
            if shapes[sp] in shapes[s].contains:
                os = os + "contains(" + str(s) + ", " + str(sp) + ");\n"
    for s in xrange(0,ns):
        for sp in xrange(0,ns):
            if s < sp and shapes[sp] in shapes[s].borders:
                os = os + "borders(" + str(s) + ", " + str(sp) + ");\n"
    return os

    
sys.setrecursionlimit(128*128)

jobs = []

for argument in sys.argv[1:]:
    if os.path.isdir(argument):
        for f in os.listdir(argument):
            if f.endswith(".png"):
                jobs = jobs + [argument + "/" + f]
    else:
        jobs = jobs + [argument]
for j in jobs:
    print j
    a = analyze(j)
    print a
    o = j[:-3]+"h"
    # useful special case
    m = re.match("svrt/results_problem_(\d+)/sample_(\d)_(\d+).png",j)
    if m:
        o = "pictures/%s_%s_%i" % (m.group(1),m.group(2),int(m.group(3)))
    with open(o,"w") as f:
        f.write(a)
    print "."


 
