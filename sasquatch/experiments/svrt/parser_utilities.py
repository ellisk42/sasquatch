import os
import numpy as np
import scipy
import scipy.misc
from scipy.misc import imrotate
import scipy.ndimage as im
import matplotlib.pyplot as plt
import pylab
import math

# a < b
def mask_subset(a,b):
    return np.sum(a > b) == 0

def mask_overlap(a,b):
    return np.any(np.logical_and(a,b))

def view(i):
    too_big = i > 255
    i = 255*too_big + i*(1-too_big)
    pylab.figure()
    pylab.axis('off')
    pylab.imshow(i,cmap = pylab.gray())
    plt.savefig('/tmp/parse_view.png')
    os.system('feh /tmp/parse_view.png')


def replace(a,old,new):
    to_replace = a == old
    return a*(1-to_replace) + new*to_replace

def neighbor_matrices(mask):
    return np.roll(mask,1,0),np.roll(mask,-1,0),np.roll(mask,1,1),np.roll(mask,-1,1)


def fill(a,x,y,o,n):
    w = a.shape[0]
    h = a.shape[1]
    if x < w and y < h and x > -1 and y > -1 and a[x,y] == o:
        a[x,y] = n
        fill(a,x-1,y,o,n)
        fill(a,x+1,y,o,n)
        fill(a,x,y-1,o,n)
        fill(a,x,y+1,o,n)

def mask_outline(m):
    o = m > False
    w = m.shape[0]
    h = m.shape[1]
    for x in range(1,w-1):
        for y in range(1,h-1):
            o[x,y] = max(m[x,y],
                         m[x-1,y],
                         m[x+1,y],
                         m[x,y-1],
                         m[x,y+1],
                         m[x+1,y+1],
                         m[x+1,y-1],
                         m[x-1,y+1],
                         m[x-1,y-1])
    return o

def center_mask(mask,mass = None):
    if mass == None:
        mass = np.sum(mask)
    w = mask.shape[0]
    h = mask.shape[1]
    xv, yv = np.meshgrid(np.arange(0,w),np.arange(0,h))
    x = int(np.sum(mask*xv)/mass)
    y = int(np.sum(mask*yv)/mass)
        
    centered = np.roll(mask,w/2-x,1)
    centered = np.roll(centered,h/2-y,0)
    
    return x,y,centered

    

def scale_mask(m,f):
    if f == 1.0:
        return m
    assert f < 1.0
    
    w = m.shape[0]
    h = m.shape[1]
    s = im.zoom(m*1.0,f,order = 0)
    
    ws = s.shape[0]
    hs = s.shape[1]
    
    _w = int((w-ws)/2)+1
    _h = int((h-hs)/2)+1
    
    k = np.zeros((w,h),dtype = bool)
    ew = _w + ws
    eh = _h + hs
    #    k[_w:ew,_h:eh] = s
    k[0:ws,0:hs] = s
    
    return center_mask(k)[2]

def rotate_mask(m,a):
    return imrotate(m,a,interp = 'nearest')

def rotation_equals(m1,m2,threshold = 3.06):
    mass = np.sum(m2)
    for a in range(360):
        r = rotate_mask(m1,a)
        differences = min([ np.sum(np.logical_xor(m2,n))
                            for n in neighbor_matrices(r) ])
        d = float(differences)/math.sqrt(mass)
        if d < threshold:
            return a
    return None

