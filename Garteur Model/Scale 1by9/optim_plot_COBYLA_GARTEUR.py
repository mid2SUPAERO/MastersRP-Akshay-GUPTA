# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:05:31 2016

@author: jmascolo
"""

import sqlitedict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

N = 5

db = sqlitedict.SqliteDict( 'modal_optim_COBYLA_GARTEUR', 'iterations' )

line_width = 2.0
label_size = 20
font_size = 20

#Create a numpy array with the number of iterations
X = np.arange(1, len(db)+1)

#Lists contaning the objective function value and constraints of all the iterations
f = []
mass = []
freq = {}

for it in db:
    f.append(db[it]['Unknowns']['f'])
    mass.append(db[it]['Unknowns']['delta_mass'])
        
for i in range(N):
    freq[i] = []
    for it in db:
        freq[i].append(db[it]['Unknowns']['delta_omega_u_'+str(i+1)])

xmin = 1
#xmax = 10

f_min = 0
f_max = np.max(f)

mass_min = min(0.,np.min(mass))
mass_max = np.max(mass)

freq_min = []
freq_max = []

for i in range(N):
    freq_min.append(min(0.,np.min(freq[i])))
    freq_max.append(np.max(freq[i]))

plt.figure(1)
plt.plot(X, f, color="green", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Objective function', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([f_min,f_max])

plt.tight_layout()
plt.savefig('objective_slsqp.pdf', bbox_inches='tight')

plt.figure(2)
plt.plot(X, mass, color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Mass Constraint', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
axes.set_ylim([mass_min,mass_max])

plt.tight_layout()
plt.savefig('total_mass_slsqp.pdf', bbox_inches='tight')

plt.figure(3)
for i in range(N):
    plt.plot(X, freq[i], color="red", linewidth=line_width)
plt.xlabel('Iterations', fontsize=font_size)
plt.ylabel('Frequency Constraint', fontsize=font_size)
plt.tick_params(labelsize=label_size)
axes = plt.gca()
#axes.set_xlim([xmin,xmax])
# axes.set_ylim([freq_min,freq_max])

plt.tight_layout()
plt.savefig('frequency_slsqp.pdf', bbox_inches='tight')
