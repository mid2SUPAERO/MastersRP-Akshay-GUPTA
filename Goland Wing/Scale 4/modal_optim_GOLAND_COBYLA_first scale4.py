# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 10:50:10 2016

@author: © Joan Mas Colomer
"""

from __future__ import print_function

from openmdao.api import Problem, Group, IndepVarComp, ExecComp, ScipyGMRES, SqliteRecorder, ScipyOptimizer, view_model

from aerostructures import NastranDynamic, ModalFunctions, DynamicStructureProblemDimensions, DynamicStructureProblemParams

#from mayavi import mlab

import numpy as np

if __name__ == "__main__":

    #Enable mode-tracking
    mode_tracking = True

    #Model with no rigid modes
    free_free = False

    #Number of normal modes to consider for the comparison
    N = 5
    #Number of normal modes to extract
    M = 10
    #Normal mode extraction method
    eigr = 'FEER'
    #Frequency lower bound (Hz)
    F1 = 0.01

    #Problem parameters (kg, mm, MPa)
    E = 1.4976E9
    nu = 0.33
    rho_s = 1.E-4
    omega_ratio = 0.5
    mass_ratio = 64.
    length_ratio = 4.

    #Baseline thickness and masses
    t_0 = np.array([.248,
                    .0096,
                    1.4224,
                    .5552])


    m_0 = np.array([125.76,
                    125.76,
                    125.76,
                    125.76,
                    62.88,
                    341.7472,
                    341.7472,
                    341.7472,
                    341.7472,
                    170.8736])

    #Design variable boundaries
#    t_max = 1.4224*np.ones(4)
#    t_min = 0.0096*np.ones(4)
    
    t_max = 5.*t_0
    t_min = 0.1*t_0

    m_max = 341.7472*np.ones(10)
    m_min = 62.88*np.ones(10)

    #Problem dimensions
    dynamic_problem_dimensions = DynamicStructureProblemDimensions()

    node_id_all = dynamic_problem_dimensions.node_id_all
    ns_all = dynamic_problem_dimensions.ns_all #number of nodes
    tn = dynamic_problem_dimensions.tn #number of thicknesses
    mn = dynamic_problem_dimensions.mn #number of masses
    sn = dynamic_problem_dimensions.sn #number of stringer sections

    #Problem parameters from Nastran model
    dynamic_problem_params = DynamicStructureProblemParams(node_id_all, N, free_free)

    #Problem parameter values
    node_coord_all = length_ratio*dynamic_problem_params.node_coord_all
    phi_ref = dynamic_problem_params.phi_ref
    eigval_ref = dynamic_problem_params.eigval_ref
    mass_ref = dynamic_problem_params.mass_ref
    omega_norm_ref = omega_ratio*np.linalg.norm(np.sqrt(eigval_ref))

    top = Problem()
    top.root = root = Group()

    #Add independent variables
    root.add('s_coord_all', IndepVarComp('node_coord_all', node_coord_all), promotes=['*'])
    root.add('Youngs_modulus', IndepVarComp('E', E), promotes=['*'])
    root.add('Poissons_ratio', IndepVarComp('nu', nu), promotes=['*'])
    root.add('material_density', IndepVarComp('rho_s', rho_s), promotes=['*'])
    root.add('thicknesses', IndepVarComp('t', np.ones(tn)), promotes=['*'])
    root.add('masses', IndepVarComp('m', np.ones(mn)), promotes=['*'])
    root.add('reference_mass', IndepVarComp('mass_ref', mass_ref), promotes=['*'])
    root.add('reference_eigvec', IndepVarComp('phi_ref', phi_ref), promotes=['*'])
    root.add('reference_eigval', IndepVarComp('eigval_ref', eigval_ref), promotes=['*'])
    root.add('frequency_ratio', IndepVarComp('omega_ratio', omega_ratio), promotes=['*'])
    root.add('model_mass_scaling', IndepVarComp('mass_ratio', mass_ratio), promotes=['*'])
    root.add('modes_number', IndepVarComp('N', float(N)), promotes=['*'])

    root.add('modal', NastranDynamic(node_id_all, tn, mn, sn, M, eigr, F1, free_free), promotes=['*'])

    root.add('mod_func', ModalFunctions(node_id_all, N, M, mode_tracking), promotes=['*'])

    root.add('obj_func', ExecComp('f=(N-MAC_trace)/N'), promotes=['*'])

    #Add mass constraints
    root.add('con_mass_upper', ExecComp('con_m_u = delta_mass'), promotes=['*'])
    root.add('con_mass_lower', ExecComp('con_m_l = delta_mass'), promotes=['*'])

    #Add frequency constraint components
    for i in range(N):
        root.add('con_freq_upper_'+str(i+1), ExecComp('delta_omega_u_'+str(i+1)+' = delta_omega['+str(i)+']', delta_omega=np.zeros(N,dtype=float)), promotes=['*'])

    for i in range(N):
        root.add('con_freq_lower_'+str(i+1), ExecComp('delta_omega_l_'+str(i+1)+' = delta_omega['+str(i)+']', delta_omega=np.zeros(N,dtype=float)), promotes=['*'])

    #Add design variable bounds as constraints - Nastran doesn't accept negative thicknesses or sections
    for i in range(tn):
        root.add('t_lower_bound_'+str(i+1), ExecComp('t_l_'+str(i+1)+' = t['+str(i)+']', t=np.zeros(tn,dtype=float)), promotes=['*'])
    for i in range(mn):
        root.add('m_lower_bound_'+str(i+1), ExecComp('m_l_'+str(i+1)+' = m['+str(i)+']', m=np.zeros(mn,dtype=float)), promotes=['*'])

    #Define solver type
    root.ln_solver = ScipyGMRES()

    #Define the optimizer (Scipy)
    top.driver = ScipyOptimizer()
    top.driver.options['optimizer'] = 'COBYLA'
    top.driver.options['disp'] = True
    top.driver.options['tol'] = 1.e-3
    top.driver.options['maxiter'] = 200
    top.driver.opt_settings['rhobeg']= 0.01

    top.driver.add_desvar('t', lower=t_min, upper=t_max, adder=-t_min, scaler=1/(t_max-t_min))

    top.driver.add_desvar('m', lower=m_min, upper=m_max, adder=-m_min, scaler=1/(m_max-m_min))

    top.driver.add_objective('f')

    scaled_mass = mass_ratio*mass_ref

    top.driver.add_constraint('con_m_u', upper=0., scaler=1/scaled_mass)
    top.driver.add_constraint('con_m_l', lower=0., scaler=1/scaled_mass)

    for i in range(N):
        top.driver.add_constraint('delta_omega_u_'+str(i+1), upper=0., scaler=1/np.sqrt(eigval_ref[i]))

    for i in range(N):
        top.driver.add_constraint('delta_omega_l_'+str(i+1), lower=0., scaler=1/np.sqrt(eigval_ref[i]))

    for i in range(tn):
        top.driver.add_constraint('t_l_'+str(i+1), lower=0., scaler=1/t_0[i])

    for i in range(mn):
        top.driver.add_constraint('m_l_'+str(i+1), lower=0., scaler=1/m_0[i])

    #Optimization Recorder
    recorder = SqliteRecorder('modal_optim_COBYLA_scale4')
    recorder.options['record_params'] = True
    recorder.options['record_metadata'] = True
    top.driver.add_recorder(recorder)

    top.setup()
    view_model(top, show_browser=False)

    #Setting initial values for design variables
    top['t'] = t_0
    top['m'] = m_0
#    top['t'] = np.array([50.,11.,10.,10.,10.,10.])
#    top['m'] = np.array([0.5,0.2,0.2])

    top.run()

    top.cleanup()  # this closes all recorders

    #Visualization
    #Points coordinates
    xs = node_coord_all

    #Eigenvectors
    ds_ref = phi_ref
    ds = root.mod_func.unknowns['ord_phi']

    #Maximum span of the model
    y_max = xs[:,1].max() - xs[:,1].min()

    #Plots
    for i in range(N):
        defs_ref = xs + 0.1*y_max*np.hstack((np.split(ds_ref[:,i:i+1], 3, 0)))
        defs = xs + 0.1*y_max*np.hstack((np.split(ds[:,i:i+1], 3, 0)))
        mlab.figure(bgcolor = (1,1,1), fgcolor = (0,0,0))
        mlab.points3d(defs_ref[:,0], defs_ref[:,1], defs_ref[:,2], color=(0,0,1), scale_factor=0.1)
        mlab.points3d(defs[:,0], defs[:,1], defs[:,2], color=(1,0,0), scale_factor=0.1)
