import os
import time
import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
import sys
import re

tiny_threshold = 15

# makes the mask bigger by one pixel around the borders
def engorge(mask):
    engorged = mask
    engorged = np.logical_or(engorged,np.roll(mask,1,0))
    engorged = np.logical_or(engorged,np.roll(mask,-1,0))
    engorged = np.logical_or(engorged,np.roll(mask,1,1))
    engorged = np.logical_or(engorged,np.roll(mask,-1,1))
    return engorged


def mass(a):
    return np.sum(a < 255)

def com(a):
    w = a.shape[0]
    h = a.shape[1]
    xv, yv = np.meshgrid(np.arange(0,w),np.arange(0,h))
    mask = a < 255
    m = np.sum(mask)
    cx = np.sum(mask*xv)/m
    cy = np.sum(mask*yv)/m
    return cx,cy

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
    
    # compute containment information
    masks = shapes < 255
    contains = np.zeros((ns,ns),dtype = np.bool_)
    for s in xrange(0,ns):
        na = np.logical_not(masks[s,:,:])
        for sp in xrange(0,ns):
            if s != sp:
                contains[s,sp] = np.sum(np.logical_and(na,masks[sp,:,:])) == 0
    # compute contact information
    border_masks = masks
    for s in xrange(0,ns):
        border_masks[s,:,:] = engorge(engorge(border_masks[s,:,:]))
        if False:
            plt.figure()
            plt.imshow(border_masks[s,:,:].astype(np.float32)*255.0,cmap=plt.cm.gray)
            plt.savefig('border'+str(s)+'.png')

    borders = np.zeros((ns,ns),dtype = np.bool_)
    for s in xrange(0,ns):
        for sp in xrange(0,ns):
            if s != sp and not contains[s,sp] and not contains[sp,s]:
                borders[s,sp] = np.sum(np.logical_and(border_masks[s,:,:],border_masks[sp,:,:])) > 0
                
    # remove tiny shapes that are touching other shapes, these are artifacts
    keep_shapes = []
    for s in xrange(0,ns):
        if mass(shapes[s,:,:]) > tiny_threshold: 
            keep_shapes.append(s)
        else:
            conflict = False
            for sp in range(ns):
                if borders[s,sp] or contains[s,sp]:
                    conflict = True
                    break
            if not conflict: keep_shapes.append(s)
    # relabel the shapes
    next_new_index = 0
    new_indexes = {}
    for s in range(ns):
        if s in keep_shapes:
            new_indexes[next_new_index] = s
            next_new_index += 1
    shapes = shapes[keep_shapes,:,:]
    ns = len(keep_shapes)
    new_contains = np.zeros((ns,ns),dtype = np.bool_)
    new_borders = np.zeros((ns,ns),dtype = np.bool_)
    for s in range(ns):
        for sp in range(ns):
            new_contains[s,sp] = contains[new_indexes[s],new_indexes[sp]]
            new_borders[s,sp] = borders[new_indexes[s],new_indexes[sp]]
    contains,borders = new_contains,new_borders

    
    centered = np.zeros((ns,w,h),dtype = np.int32)
    xs = [0]*ns
    ys = [0]*ns
    for s in xrange(0,ns):
        cx,cy = com(shapes[s,:,:])
        ys[s] = int(cy)
        xs[s] = int(cx)
        centered[s,:,:] = np.roll(shapes[s,:,:],w/2-cx,1)
        centered[s,:,:] = np.roll(centered[s,:,:],h/2-cy,0)
    
    # identify identical shapes
    kind = [0]
    for s in xrange(1,ns):
        is_duplicate = False
        for sp in xrange(0,s):
            d = np.logical_xor(centered[s,:,:],centered[sp,:,:])
            if np.sum(d) < 5:
                kind = kind + [kind[sp]]
                is_duplicate = True
                break
        if not is_duplicate:
            kind = kind + [max(kind)+1]
    
    # build output string
    os = ""
    for s in xrange(0,ns):
        os = os + "[" + str(xs[s]) + ", " + str(ys[s]) + ", " + str(mass(shapes[s,:,:])) + ", " + str(kind[s])+"]"
        if s < ns-1: os = os + ", "
    os = os + "\n"
    for s in xrange(0,ns):
        for sp in xrange(0,ns):
            if s != sp and contains[s,sp]:
                os = os + "contains(" + str(s) + ", " + str(sp) + ");\n"
    for s in xrange(0,ns):
        for sp in xrange(0,ns):
            if s != sp and borders[s,sp]:
                os = os + "borders(" + str(s) + ", " + str(sp) + ");\n"
    return os

    
sys.setrecursionlimit(128*128)

jobs = []

if os.path.isdir(sys.argv[1]):
    for f in os.listdir(sys.argv[1]):
        if f.endswith(".png"):
            jobs = jobs + [sys.argv[1] + "/" + f]
else:
    jobs = jobs + [sys.argv[1]]
for j in jobs:
    print j
    a = analyze(j)
    print a
    o = j[:-3]+"h"
    # useful special case
    m = re.match("results_problem_(\d+)/sample_(\d)_(\d+).png",j)
    if m:
        o = "pictures/%s_%s_%i" % (m.group(1),m.group(2),int(m.group(3)))
    with open(o,"w") as f:
        f.write(a)
    print "."


 
