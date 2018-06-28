# -*- coding: utf-8 -*-
"""
Created on Wed Sep 07 16:13:13 2016

@author: jmascolo
"""

import numpy as np

import math

from mayavi import mlab

user_input = raw_input('Enter the name of the Nastran file, please')

input_file = user_input + '.inp'

output_file = user_input + '.pnh'

#Dictionary containing grid point coordinates
coord = {}

#Dictionary containing modal displacements
modes = {}

#Dictionary containing eigenvalues
eigval = {}

#Number of modes
N = 0

#Read grid point coordinates from input file
with open(input_file) as f:
    lines = f.readlines()
    lines = [i.split(',') for i in lines]
    
    for line in lines:
        if len(line) > 1:
            if line[0] == 'GRID':
                coord[int(line[1])] = [float(line[3]), float(line[4]), float(line[5])]

#Read modal displacements from .pnh file
with open(output_file) as f:
    lines = f.readlines()
    lines = [i.split() for i in lines]
    
    for line in lines:
        if len(line) > 1:
            if line[0] == '$EIGENVALUE':
                N = int(line[5])
                eigval[N] = float(line[2])
            elif line[1] == 'G':
                #First number of the tuple indicates mode number
                modes[(N, int(line[0]))] = [float(line[2]), float(line[3]), float(line[4])]
                
xs = []
for item in coord:
    xs.append([coord[item][0], coord[item][1], coord[item][2]])

xs = np.asarray(xs)
max_xs = xs[:,0].max() - xs[:,0].min()
max_ys = xs[:,1].max() - xs[:,1].min()
max_zs = xs[:,2].max() - xs[:,2].min()
max_l = max(max_xs, max_ys, max_zs)

for i in range(1, 11):
    ds = []
    
    for item in coord:
        ds.append([modes[(i, item)][0], modes[(i, item)][1], modes[(i, item)][2]])        
        
    ds = np.asarray(ds)
    
    max_xd = ds[:,0].max() - ds[:,0].min()
    max_yd = ds[:,1].max() - ds[:,1].min()
    max_zd = ds[:,2].max() - ds[:,2].min()
    max_d = max(max_xd, max_yd, max_zd)
    
    ds = 0.15*max_l/max_d*ds
    
    defs = xs + ds
    
    f = np.sqrt(abs(eigval[i]))/(2*math.pi)
    
    mlab.figure(bgcolor = (1,1,1), fgcolor = (0,0,0))
    mlab.points3d(defs[:,0], defs[:,1], defs[:,2], color=(0,0,1), scale_factor=0.005*max_l)
    mlab.title('Mode '+str(i)+', f = '+"{:.2f}".format(f)+' Hz')