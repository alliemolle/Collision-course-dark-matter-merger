# -*- coding: utf-8 -*-
"""
Allie Molle

collision course
PHYS 410

changing small params
"""

# Import modules ==============================================================
import numpy as np, timeit

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

import cc_module as cc

# Main script =================================================================

# Physical constants
G = 4.5e-6              # grav constant [kpc^3/M_sun*Gyr^2]
rho_crit = 136          # critical background density z = 0 [M_sun/kpc^3]
delt = 337              # overdensity threshold of virial radius
cvir0 = 10              # initial concentrations

# Simulation constants
theta = 0.9             # Barnes-Hut minimum distance ratio L/D
soft_len = 0.001        # 1pc Softening length [kpc]
max_particles = 10      # Max particles per leaf node
min_hw = soft_len       # minimum half-width for octree nodes
virial_tol = 1e-2       # threshold for virialization
nfit = 100              # number of bins to make density profile

dt = 0.005              # 500 yr timestep [Gyr]
Time = 9                # maximum run time [Gyr]
plot_every = 10         # how often plot
plot_size = [1,2,3]     # size of frame for watching simulation

# create energy + virials tracking arrays
n_step = int(Time/dt)                   # number of timesteps
K = np.zeros(n_step, dtype=np.float64)  # kinetic energy
U = np.zeros(n_step, dtype=np.float64)  # potential energy
virials = np.zeros(n_step)              # relative error from virialization
cvir = np.zeros((n_step,3))             # cvir values from all three sims
Q_vals = np.zeros((n_step,3))           # Q vals from all three sims

# Merger sizes [1=major,10=minor]
mu = [1,5,10]
merger = ['major', 'mid', 'minor']
c_host = ['lightcoral','m','blueviolet']

for j in range(3):
        
    # create satellite halo -------------------------------------------------------
    M_sat = 1e4                 # total mass in halo
    N_sat = int(1e3)            # number of particles 
    m_particle = M_sat/N_sat    # mass per particle
    print(f'particle mass = {m_particle:.2f}')
    
    # halo internal structure: virial radius, scale radius, and density at r_scale
    rvirs, r_scale, rho_scale = cc.internal_halo_structure(M_sat,delt,rho_crit,cvir0)
    
    # positions and velocities of particles
    pos1, vel1 = cc.halo_init(rvirs, r_scale, rho_scale, N_sat, soft_len, G, seed=12345678)
    
    # create host halo ------------------------------------------------------------
    M_host = M_sat*mu[j]        
    N_host = int(M_host/m_particle)
    
    # halo internal structure: virial radius, scale radius, and density at r_scale
    rvirh, r_scale, rho_scale = cc.internal_halo_structure(M_host,delt,rho_crit,cvir0)
    
    # positions and velocities of particles
    pos2, vel2 = cc.halo_init(rvirh, r_scale, rho_scale, N_host, soft_len, G, seed=87654321)
    
    # combine halo arrays and place in sim ----------------------------------------
    
    # move halos to positions (x_init,0,0) and (-x_init,0,0)
    x_init = rvirh
    target1 = np.array([x_init, 0.00, 0.00])
    target2 = np.array([-x_init, 0.00, 0.00])
    pos1_placed = cc.recenter_positions(pos1, m_particle, target1)
    pos2_placed = cc.recenter_positions(pos2, m_particle, target2)
    
    pos = np.vstack((pos1_placed,pos2_placed))
    vel = np.vstack((vel1,vel2))
    
    # plot initial positions ------------------------------------------------------
    colors = np.array(['c']*len(pos1) + [c_host[j]]*len(pos2))
    
    # find initial acceleration tree to start simulation
    acc = cc.compute_acc_tree_numba(pos, m_particle,theta, soft_len, max_particles,min_hw,G)
    
    # Initial energies BEFORE loop (t=0)
    K[0] = cc.kinetic_energy_numba(vel,m_particle)
    U[0] = cc.potential_energy_numba(pos,m_particle, G, soft_len)
    
    # Check if system is bound
    if abs(U[0]) > K[0]:
        print("  System is BOUND (|PE| > KE): will COLLAPSE")
    else:
        print("  System is UNBOUND (|PE| < KE): will EXPAND")
    
    # loop through time -----------------------------------------------------------
    start = timeit.default_timer()
    for i in range(n_step):
        
        t = i*dt  # current time
        pos, vel, acc = cc.leapfrog_step_numba(pos, vel, acc, dt, m_particle,theta, soft_len, max_particles, min_hw,G)
        
        # get position and velocity rel to CM
        CM = np.mean(pos, axis=0) 
        r = np.linalg.norm(pos-CM,axis=1)
        
        # filter out unbound particle from cvir calculation (Ki > Ui)
        Upart = -G * m_particle * pos.shape[0] / (r + soft_len)
        v2 = np.sum(vel**2, axis=1)
        Kpart = 0.5 * m_particle * v2
        Epart = Kpart + Upart
        bound_mask = Epart < 0.0
        
        pos_bound = pos[bound_mask]
        vel_bound = vel[bound_mask]
        
        # find system K and U
        K[i] = cc.kinetic_energy_numba(vel,m_particle)
        U[i] = cc.potential_energy_numba(pos,m_particle,G, soft_len)
        
        virials[i] = (2*K[i] + U[i]) / abs(U[i])
        
        # Plot 3D structure occasionally to make movie 
        if i % plot_every == 0:
            # 3D plot pos
            fig = cc.Plothalo_3D(pos,t,colors,plot_size[j], merger[j])
            #fig.savefig(f'merger_movies/{merger[j]}/{merger[j]}_t{t:.2f}.png', dpi=300)
            plt.show()
            plt.close(fig)
            
        # after halos collide, calc cvir every 0.5 Gyr
        if t % 1 == 0:
            
            # find maximum bound particle radius = rvir 
            r_bound = r[bound_mask]
            rvir = np.max(r_bound)
    
            # Create radial bins + use histogram to create a density profile
            rgrid = np.logspace(np.log10(soft_len), np.log10(rvir), nfit+1)
            counts,edges = np.histogram(r_bound, bins=rgrid)
            mass_per_bin = counts*m_particle
    
            # find volumes to convert mass per bin to density per bin
            r_in = edges[:-1]
            r_out = edges[1:]
            r_centers = np.sqrt(r_in * r_out)      # geometric mean for log bins
            shell_vols = (4.0/3.0) * np.pi * (r_out**3 - r_in**3)
    
            # find densities + report any empty shells
            density = mass_per_bin / shell_vols
            if np.any(density) < 1e-2:
                print(f'density = {density} M☉/kpc**3')
    
            # create array of lower bounds to try fitting and call fit func
            rs_bounds = np.linspace(soft_len, rvir/cvir0, 3)
            fit, Q = cc.curvefit_NFW_mass(r_centers, mass_per_bin, 1e2, 1e6, soft_len, rvir)
    
            # use best fit parameters to find cvir
            rhos = fit[0]
            rs = fit[1]
            cvir[i,j] = rvir/rs
            Q_vals[i,j] = Q
            
            fit_check = cc.NFW_prof(r_centers,rhos,rs)
            chi2 = np.sum((density - fit_check)**2 / fit_check)
    
            print(f'INFO: cvir value = {cvir[i,j]:.3f} Q = {Q:.3f}  t={t:.2f}')
    
            plt.figure(figsize=(10, 6))
            plt.scatter(r_centers, density, s=15,c='r',label='Sampled particles', alpha=0.6)
            plt.loglog(r_centers, fit_check, '-',c='b', label='NFW fit', alpha=0.6)
            plt.xlabel('Radius [kpc]')
            plt.ylabel('Density [M☉/kpc³]')
            plt.legend()
            plt.title(f'NFW fit {merger[j]} merger: t={t:.2f}')
            plt.grid(True, alpha=0.3)
            #plt.savefig(f'collision_course/NFWfit_{merger[j]}_t{t:.2f}.png')
            plt.show()
            plt.close()
            
            # if hasn't merged yet, retain og cvir
            if t < 2.0:
                cvir[i,j] = cvir0
                Q_vals[i,j] = 0.0

            
    end = timeit.default_timer()
    print(f'time to run sim: {end - start}')
        
    # plot energies over time
    t = np.linspace(0,Time,num=n_step)
    plt.figure(dpi=300)
    plt.plot(t, (K+U)/K, 'm', label = 'total energy')
    plt.plot(t, 2*K +U, 'c', label = 'virialization')
    plt.xlabel('time [yr]')
    plt.ylabel('energy [M☉*kpc$^2$/Gyr$^2$]')
    plt.title(f'Energy of {merger[j]} merger over time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    #plt.savefig(f'collision_course/energies_{merger[j]}.png')
    plt.show()
    plt.close()
    
    
    # plot cvir over time
    cvir_mask = cvir[:,j] != 0
    tc = t[cvir_mask]
    cvir_plt = cvir[:,j][cvir_mask]
    Q_vals_plt = Q_vals[:,j][cvir_mask]
    plt.figure()
    plt.errorbar(tc, cvir_plt, yerr=Q_vals_plt,color=c_host[j], fmt='o-', ecolor=c_host[j],capsize=3)

    plt.xlabel('time (yr)')
    plt.ylabel('cvir')
    plt.title(f'Cvir of {merger[j]} merger over time')
    plt.grid(True, alpha=0.3)
    #plt.savefig(f'collision_course/cvir_{merger[j]}.png')
    plt.show()
    plt.close()
    
    

# plot cvir over time for all mergers
plt.figure(dpi=300)
for j in range(3):
    # plot cvir over time 
    cvir_mask = cvir[:,j] != 0
    tc = t[cvir_mask]
    cvir_plt = cvir[:,j][cvir_mask]
    Q_vals_plt = Q_vals[:,j][cvir_mask]
    plt.errorbar(tc, cvir_plt, yerr=Q_vals_plt,color=c_host[j], fmt='o-', ecolor=c_host[j],capsize=3)
plt.xlabel('time (yr)')
plt.ylabel('cvir')
plt.title('Cvir of mergers over time')
plt.grid(True, alpha=0.3)
#plt.savefig('collision_course/cvir_all.png')
plt.show()
plt.close()

