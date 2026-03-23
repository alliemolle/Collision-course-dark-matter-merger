# -*- coding: utf-8 -*-
"""
Allie Molle
PHYS 410

collision course:
    functions module
"""
#%% Import modules 
import numpy as np
from numba import njit

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
plt.rcParams.update(plt.rcParamsDefault)

#%% Halo Initialization
def halo_init(rvir, r_scale, rho_scale,N,soft_len,G,seed=12345678):
    """
    Creates array of initial positions and velocities 
    for DM-only halo
    Inputs (all floats): 
            rvir: virial radius [kpc]
            r_scale: scale radius [kpc]
            rho_scale: density at r_scale [M_sun/kpc^3]
            N: number of particles in halo
            soft_len: softening length [kpc]
            G: grav const [kpc^3/M_sun*Gyr^2]
    Returns: 
            positions: 2D numpy array (Nx3)
                       particles positions (xi,yi,zi)
            velocities: 2D numpy array (Nx3)
                        particle velocities (vxi,vyi,vzi)
    """
    # define seed for random generator
    rng = np.random.default_rng(seed=seed)

    # init polar coords
    theta = rng.random(N)*np.pi
    phi = rng.random(N)*2*np.pi
    r = radial_pos(rvir,r_scale,rho_scale,N,soft_len,rng)
    
    N = r.shape[0]
    pos = np.zeros((N, 3), dtype=np.float64)
    vel = np.zeros((N, 3), dtype=np.float64)
    
    # find initial positions and velocities 
    for i in range(N):
        ri = r[i]
        M_enc = mass_enclosed_NFW(ri, rho_scale, r_scale)
        
        # ensure no divide by zero
        r_safe = ri if ri > 0.0 else 1e-12
        Vc = np.sqrt(G * M_enc / r_safe)

        sinth = np.sin(theta[i])
        ph = phi[i]

        pos[i, 0] = ri * sinth * np.cos(ph)
        pos[i, 1] = ri * sinth * np.sin(ph)
        pos[i, 2] = ri * np.cos(theta[i])

        vel[i, 0] = -pos[i, 1] * (Vc / r_safe)
        vel[i, 1] =  pos[i, 0] * (Vc / r_safe)
        vel[i, 2] = 0.0

    return pos, vel


def radial_pos(rvir, r_scale, rho_scale, N, soft_len, rng):
    """
    Create radial distribution for N particles to model NFW profile in 
    spherically symmetric halo by using CDF interpolation + NFW params
    Inputs:
            [0,4] explained in halo_init
            rng: rng object created in halo_init
    Returns:
           radii: 1D numpy array (N)
                   particles radial positions [kpc]
    """
    
    # Create radial grid
    r_grid = np.logspace(np.log10(soft_len), np.log10(rvir), 1000)
    
    # Calculate CDF: M(<r) / M_total in each 'bin' 
    M_grid = mass_enclosed_NFW_arr(r_grid, rho_scale, r_scale)
    M_total = M_grid[-1]
    CDF = M_grid / M_total

    # Sample uniform random numbers [0, 1]
    u = rng.uniform(0, 1, N)
    
    # Interpolate to find radii
    radii = np.interp(u, CDF, r_grid)
    
    return radii

def mass_enclosed_NFW_arr(r, rho_scale, r_scale):
    """
    Analytical integral solution for NFW profile
        M(<r) = integral(4pi r^2 rho_NFW(r) dr) for [0,r]
    """
    x = r / r_scale
    return 4.0 * np.pi * rho_scale * r_scale**3 * (np.log(1.0 + x) - x / (1.0 + x))

def circular_velocity(r,r_scale,rho_scale,N,G):
    """
    Finds circular velocities at radii in array r
    Inputs:
            explained in halo_init
    Returns:
            N length array of circular velocities of the particles [kpc/Gyr]
            (will be magnitudes with no direction)
    """
    M_enc = np.zeros(N, dtype=np.float64)

    for i in range(N):
        M_enc[i] = mass_enclosed_NFW(r[i], r_scale, rho_scale)
        if M_enc[i] < 0: M_enc[i] = 0
        if r[i] < 1e-8: r[i] = 1e-8
    return np.sqrt(G*M_enc /r)

def recenter_positions(pos, m_particle, target_point):
    """
    Shift particle positions so that the center of mass (COM) 
    of the system is at the given target_point
    Inputs:
            pos: 2D array of floats (Nx3)
                 particle positions [xi,yi,zi]
            m_particle: mass of single particle
            target_point: 1D array of floats (3)
    Returns: new particles positions in form 'pos'       
    """
    # Compute current COM
    com = np.mean(pos, axis=0)  # since all particles same mass
    # Compute offset
    offset = target_point - com
    # Shift positions
    pos_new = pos + offset
    return pos_new

def internal_halo_structure(Mvir,delt,rho_crit,cvir0):
    """
    Find rvir for a halo of size Mvir at z=0
    Find r_scale for a given initial concentration
    Find rho_scale for the halo given an NFW profile 
    Parameters:
            Mvir: virial mass (M_sun)
            delt: virial overdensity parameter
            rho_crit: critical background density at z=0 (M_sun/kpc^3)
            cvir0: initial concentration 
    """
    rvir = (3*Mvir/(4*np.pi*delt*rho_crit))**(1/3)
    r_scale = rvir / cvir0
    x = cvir0
    f_x = np.log(1 + x) - x/(1 + x)
    rho_scale = Mvir / (4*np.pi*r_scale**3*f_x)
    return rvir, r_scale, rho_scale

#%% Numba kernels: energies, leapfrog, octree ====

@njit
def kinetic_energy_numba(vel, m_particle):
    """
    Find kinetic energy of system of particles: K = sum(mi v^2)/2
    Inputs:
        vel: 2D array of floats (Nx3)
             particle velocities [xi,yi,zi]
        m_particle: mass of single particle
    """
    N = vel.shape[0]
    K = 0.0
    for i in range(N):
        v2 = vel[i,0]*vel[i,0] + vel[i,1]*vel[i,1] + vel[i,2]*vel[i,2]
        K += 0.5 * m_particle * v2
    return K
@njit
def potential_energy_numba(pos, m_particle, G, soft_len=0.0):
    """
    Find pairwise potential energy of system with only grav force:
            U = - sum_{i<j} G m^2 / r_ij
    Inputs:
            pos: 2D array of floats (Nx3)
                 particle positions [xi,yi,zi]
            m_particle: mass of single particle
            
    """
    N = pos.shape[0]
    U = 0.0
    for i in range(N-1):
        xi = pos[i,0]; yi = pos[i,1]; zi = pos[i,2]
        for j in range(i+1, N):
            dx = pos[j,0] - xi
            dy = pos[j,1] - yi
            dz = pos[j,2] - zi
            r = np.sqrt(dx*dx + dy*dy + dz*dz + soft_len*soft_len)
            U -= G * m_particle * m_particle / r
    return U

@njit
def mass_enclosed_NFW(r, rho_scale, r_scale):
    """
    Analytical integral solution for NFW profile
        M(<r) = integral(4pi r^2 rho_NFW(r) dr) for [0,r]
    """
    x = r / r_scale
    return 4.0 * np.pi * rho_scale * r_scale**3 * (np.log(1.0 + x) - x / (1.0 + x))


@njit
def build_flat_octree(pos, m_particle, max_particles, min_hw, max_nodes, max_pidx):
    """
    author: Allie's octree code + chatgpt + prompt 'please rewrite code such that
            it will perform the exact same functions in a numba compatible way
    
    Produces flat octree with loops and large arrays: faster, but worse for 
    memory allocation 
    Inputs:
            pos: 2D numpy array (Nx3)
                 particle positions (xi,y,zi) [kpc]
            m_particle: mass of single particle [M_sun]
            max_particles: maximum particles that can be in a leaf 
            min_hw: minimum half-width for octree nodes [kpc]
            max_nodes: maximum number of nodes that can be created (prevent 
                                                                    buffering)
            max_pidx: max
    Returns:
            node_center: 2D numpy array (max_nodes x 3)
                         center coordinates of each node [kpc]
            node_hw: 1D numpy array (max_nodes)
                     half-width of each node [kpc]
            node_mass: 1D numpy array (max_nodes)
                       total mass in each node [M_sun]
            node_com: 2D numpy array (max_nodes x 3)
                      center of mass of each node [kpc]
            node_child: 2D numpy array (max_nodes x 8)
                        indices of 8 child nodes (-1 if no child)
            node_start: 1D numpy array (max_nodes)
                        starting index in particle_index for leaf nodes (-1 for internal)
            node_count: 1D numpy array (max_nodes)
                        number of particles in leaf nodes (0 for internal)
            particle_index: 1D numpy array (max_pidx)
                            reorganized particle indices grouped by node
            num_nodes: int
                       actual number of nodes created in the tree
    """
    N = pos.shape[0]
    # bounding box
    pos_max = 0.0
    for i in range(N):
        # use absolute max coordinate
        for d in range(3):
            val = pos[i,d]
            if val < 0:
                val = -val
            if val > pos_max:
                pos_max = val

    center0 = np.zeros(3, dtype=np.float64)
    half_width0 = pos_max * 1.1 if pos_max > 0.0 else 1.0

    # allocate arrays
    node_center = np.zeros((max_nodes, 3), dtype=np.float64)
    node_hw = np.zeros(max_nodes, dtype=np.float64)
    node_mass = np.zeros(max_nodes, dtype=np.float64)
    node_com = np.zeros((max_nodes, 3), dtype=np.float64)
    node_child = -1 * np.ones((max_nodes, 8), dtype=np.int64)
    node_start = -1 * np.ones(max_nodes, dtype=np.int64)
    node_count = np.zeros(max_nodes, dtype=np.int64)
    particle_index = -1 * np.ones(max_pidx, dtype=np.int64)

    # initialize root
    num_nodes = 1
    node_center[0,0] = center0[0]; node_center[0,1] = center0[1]; node_center[0,2] = center0[2]
    node_hw[0] = half_width0

    # root particle indices in particle_index[0:N]
    for i in range(N):
        particle_index[i] = i
    node_start[0] = 0
    node_count[0] = N

    # compute root mass & com
    total_mass = 0.0
    comx = 0.0; comy = 0.0; comz = 0.0
    for ii in range(N):
        idx = particle_index[ii]
        total_mass += m_particle
        comx += pos[idx,0]
        comy += pos[idx,1]
        comz += pos[idx,2]
    if total_mass > 0.0:
        node_mass[0] = total_mass
        node_com[0,0] = comx / (node_count[0])
        node_com[0,1] = comy / (node_count[0])
        node_com[0,2] = comz / (node_count[0])
    else:
        node_mass[0] = 0.0

    # iterative build using a simple queue index
    node_idx = 0
    next_free_pidx = N  # next free position in particle_index buffer
    while node_idx < num_nodes:
        count = node_count[node_idx]
        hw = node_hw[node_idx]
        if count <= max_particles or hw < min_hw:
            # leaf, do not subdivide
            node_idx += 1
            continue

        # create 8 temporary buckets for indices
        child_counts = np.zeros(8, dtype=np.int64)
        # first pass: count how many particles go to each child
        start = node_start[node_idx]
        cx = node_center[node_idx,0]; cy = node_center[node_idx,1]; cz = node_center[node_idx,2]
        half = hw * 0.5

        for p_i in range(start, start + count):
            idx = particle_index[p_i]
            x = pos[idx,0]; y = pos[idx,1]; z = pos[idx,2]
            code = 0
            if x > cx: code += 1
            if y > cy: code += 2
            if z > cz: code += 4
            child_counts[code] += 1

        # second pass: allocate child nodes and write indices into particle_index
        child_offsets = np.zeros(8, dtype=np.int64)
        # compute offsets (cumulative)
        offset_cursor = next_free_pidx
        for c in range(8):
            child_offsets[c] = offset_cursor
            offset_cursor += child_counts[c]
        # check not overflow
        if offset_cursor > max_pidx:
            # out of preallocated space; stop subdividing further (make leaf)
            node_idx += 1
            continue

        # Now copy indices into their child slots and create child nodes
        # We'll need temporary counters to place each index
        cur_counts = np.zeros(8, dtype=np.int64)
        # initialize new children placeholders to -1
        new_child_indices = -1 * np.ones(8, dtype=np.int64)

        for p_i in range(start, start + count):
            idx = particle_index[p_i]
            x = pos[idx,0]; y = pos[idx,1]; z = pos[idx,2]
            code = 0
            if x > cx: code += 1
            if y > cy: code += 2
            if z > cz: code += 4

            dest = child_offsets[code] + cur_counts[code]
            particle_index[dest] = idx
            cur_counts[code] += 1

        # For each child with >0 particles, create a new node
        for c in range(8):
            if child_counts[c] > 0:
                if num_nodes >= max_nodes:
                    # out of nodes: skip subdividing this node (make leaf)
                    child_counts[c] = 0
                    continue
                child_idx = num_nodes
                num_nodes += 1
                new_child_indices[c] = child_idx

                # compute child's center offset
                dx = 1.0 if (c & 1) else -1.0
                dy = 1.0 if (c & 2) else -1.0
                dz = 1.0 if (c & 4) else -1.0
                node_center[child_idx,0] = cx + dx * (half / 2.0)
                node_center[child_idx,1] = cy + dy * (half / 2.0)
                node_center[child_idx,2] = cz + dz * (half / 2.0)
                node_hw[child_idx] = half

                # assign particle indices start/count
                node_start[child_idx] = child_offsets[c]
                node_count[child_idx] = child_counts[c]

                # compute mass and COM for child
                smx = 0.0; smy = 0.0; smz = 0.0
                for kk in range(node_start[child_idx], node_start[child_idx] + node_count[child_idx]):
                    pid = particle_index[kk]
                    smx += pos[pid,0]; smy += pos[pid,1]; smz += pos[pid,2]
                node_mass[child_idx] = child_counts[c] * m_particle
                node_com[child_idx,0] = smx / (node_count[child_idx])
                node_com[child_idx,1] = smy / (node_count[child_idx])
                node_com[child_idx,2] = smz / (node_count[child_idx])

        # attach children (or -1)
        for c in range(8):
            node_child[node_idx, c] = new_child_indices[c]

        # mark parent count = 0 (we will not use its particle list anymore)
        node_count[node_idx] = 0
        node_start[node_idx] = -1

        # advance next_free_pidx
        next_free_pidx = offset_cursor
        node_idx += 1

    # finished building; return arrays and num_nodes and particle_index buffer
    return (node_center, node_hw, node_mass, node_com, node_child,
            node_start, node_count, particle_index, num_nodes)


@njit
def compute_acc_from_tree(pos, node_center, node_hw, node_mass, node_com,
                          node_child, node_start, node_count, particle_index,
                          num_nodes, theta, soft_len, m_particle, G):
    """
    author: Allie's octree code + chatgpt + prompt 'please rewrite code such that
            it will perform the exact same functions in a numba compatible way
            
    Computes gravitational accelerations using Barnes-Hut algorithm
    with a prebuilt flat octree structure.

    Inputs:
            [0,9] explained in build_flat_octree
            [10,13] explained in compute_acc_tree_numba
    Returns:
            acc: 2D numpy array (Nx3)
                particle accelerations (axi, ayi, azi) [kpc/Gyr^2]
    """
    N = pos.shape[0]
    acc = np.zeros((N, 3), dtype=np.float64)
    # temporary stack for node traversal
    stack = -1 * np.ones(1024, dtype=np.int64)  # stack of node indices; depth-limited (1024)
    for i in range(N):
        # push root
        top = 0
        stack[top] = 0
        ax = 0.0; ay = 0.0; az = 0.0
        xi = pos[i,0]; yi = pos[i,1]; zi = pos[i,2]

        while top >= 0:
            node = stack[top]; top -= 1
            # skip invalid node
            if node < 0 or node >= num_nodes:
                continue

            nmass = node_mass[node]
            if nmass == 0.0:
                continue

            # compute r_vec from target to node COM
            rx = node_com[node,0] - xi
            ry = node_com[node,1] - yi
            rz = node_com[node,2] - zi
            dist = np.sqrt(rx*rx + ry*ry + rz*rz)
            # if leaf (has particle list)
            if node_start[node] >= 0 and node_count[node] > 0:
                # direct-sum over particles in this leaf
                start = node_start[node]
                cnt = node_count[node]
                for pidx in range(start, start + cnt):
                    j = particle_index[pidx]
                    if j == i:
                        continue
                    dx = pos[j,0] - xi
                    dy = pos[j,1] - yi
                    dz = pos[j,2] - zi
                    r2 = dx*dx + dy*dy + dz*dz + soft_len*soft_len
                    invr3 = 1.0 / (r2 * np.sqrt(r2))
                    ax += G * m_particle * dx * invr3
                    ay += G * m_particle * dy * invr3
                    az += G * m_particle * dz * invr3
            else:
                # internal node or empty; decide whether to use multipole or open
                box_size = 2.0 * node_hw[node]
                if dist == 0.0 or (box_size / dist) < theta:
                    # use node's COM approximation
                    r2 = dist*dist + soft_len*soft_len
                    invr3 = 1.0 / (r2 * np.sqrt(r2))
                    ax += G * node_mass[node] * rx * invr3
                    ay += G * node_mass[node] * ry * invr3
                    az += G * node_mass[node] * rz * invr3
                else:
                    # push children
                    for c in range(8):
                        child = node_child[node, c]
                        if child != -1:
                            top += 1
                            if top < stack.shape[0]:
                                stack[top] = child
                            else:
                                # stack overflow: fall back to direct evaluate this child immediately
                                # (rare; only if tree very deep)
                                # We'll evaluate child directly by placing it into acc terms via recursion-like behavior
                                # but because recursion isn't allowed, we'll evaluate its particles if leaf or push grandchildren.
                                # For simplicity, skip pushing if overflow (rare); could be improved)
                                pass
        acc[i,0] = ax
        acc[i,1] = ay
        acc[i,2] = az
    return acc

@njit
def compute_acc_tree_numba(pos, m_particle, theta, soft_len, max_particles, min_hw, G):
    """
    Build flat octree and compute accelerations via Barnes-Hut (all in njit).
    Inputs:
           pos: 2D numpy array (Nx3)
                particle positions (xi,y,zi) [kpc]
           m_particle: mass of single particle [M_sun]
           theta: Barnes-Hut minimum distance ratio L/D
           soft_len: softening length [kpc]
           max_particles: Max particles per leaf node
           min_hw: min
    """
    N = pos.shape[0]
    # heuristics for preallocation
    max_nodes = 8 * N if 8 * N > 64 else 64
    max_pidx = 8 * N if 8 * N > 64 else 64

    (node_center, node_hw, node_mass, node_com, node_child,
     node_start, node_count, particle_index, num_nodes) = build_flat_octree(pos, m_particle, max_particles, min_hw, max_nodes, max_pidx)

    acc = compute_acc_from_tree(pos, node_center, node_hw, node_mass, node_com,
                                node_child, node_start, node_count, particle_index,
                                num_nodes, theta, soft_len, m_particle, G)
    return acc

# ---------- Leapfrog wrapper that uses numba octree compute -----------------
@njit
def leapfrog_step_numba(pos, vel, acc, dt, m_particle, theta, soft_len, max_particles, min_hw, G):
    # vel half-step
    N = pos.shape[0]
    vel_half = np.empty((N,3), dtype=np.float64)
    for i in range(N):
        vel_half[i,0] = vel[i,0] + 0.5 * dt * acc[i,0]
        vel_half[i,1] = vel[i,1] + 0.5 * dt * acc[i,1]
        vel_half[i,2] = vel[i,2] + 0.5 * dt * acc[i,2]

    pos_new = np.empty((N,3), dtype=np.float64)
    for i in range(N):
        pos_new[i,0] = pos[i,0] + dt * vel_half[i,0]
        pos_new[i,1] = pos[i,1] + dt * vel_half[i,1]
        pos_new[i,2] = pos[i,2] + dt * vel_half[i,2]

    acc_new = compute_acc_tree_numba(pos_new, m_particle, theta, soft_len, max_particles, min_hw, G)

    vel_new = np.empty((N,3), dtype=np.float64)
    for i in range(N):
        vel_new[i,0] = vel_half[i,0] + 0.5 * dt * acc_new[i,0]
        vel_new[i,1] = vel_half[i,1] + 0.5 * dt * acc_new[i,1]
        vel_new[i,2] = vel_half[i,2] + 0.5 * dt * acc_new[i,2]

    return pos_new, vel_new, acc_new

#%% NFW fitting Functions 

def NFW_prof(r, rho_s, r_s):
    """
    NFW profile
    Parameters:
        r     : 1D array of floats
                radius
        rho_s : float
                density at scale radius
        r_s   : float
                scale radius
    """
    return rho_s/((r/r_s)*(1 + (r/r_s))**2)

def shell_mass_NFW(r,rho_s, r_s):
    """
    Finds mass in shells defined by radii array using NFW density profile
    distribution
    Inputs:
            r: 1D array (nfit+1)
               radial edges of mass shells 
            rho_s: density at scale radius
            r_s: scale radius
    Returns:
            M_shell: 1D array (nfit)
                     mass in each shell
    """
    r = np.copy(r)
    dr = r[1] - r[0]
    r_in  = r - 0.5 * dr
    r_out = r + 0.5 * dr
    
    M_shell = mass_enclosed_NFW(r_out, rho_s, r_s) - mass_enclosed_NFW(r_in, rho_s, r_s)
    return M_shell


def curvefit_NFW_mass(r_centers, mass_per_bin, rho0_l, rho0_h, rs_l, rs_h):
    """
    Fits a mass profile of spherical shells to NFW density profile with bounded
    parameters rho_scale and r_scale 
    Inputs:
            r_centers: 1D numpy array (n_shells) 
                       median radii in each shell 
            mass_per_bin: 1D numpy array (n_shells)
                          mass in each shell
            rho0_l, rho0_h: bounds for rho_scale
            rs_l, rs_h: bounds for r_scale
    Returns:
            fit: 1D array (2)
                 fit parameters [rho_scale,r_scale]
            Q: goodness of fit parameter
    """

    # convert shell masses to enclosed masses
    M_data = np.cumsum(mass_per_bin)
    
    # remove empty bins
    valid = M_data > 0
    r_centers = r_centers[valid]
    M_data = M_data[valid]

    bounds = ([rho0_l, rs_l], [rho0_h, rs_h])

    fit, cov = curve_fit(mass_enclosed_NFW_arr, r_centers, M_data, bounds=bounds)

    M_model = mass_enclosed_NFW_arr(r_centers, *fit)
    Q = np.sqrt(np.mean(((M_data - M_model)/M_model)**2))

    return fit, Q

#%% Plotting function

def Plothalo_3D(pos,t,colors,plot_size, merger):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=10, azim=60) 
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], c=colors,s=10, alpha=0.75)
    ax.set_xlim(-plot_size, plot_size)   # x-axis limits
    ax.set_ylim(-plot_size, plot_size)   # y-axis limits
    ax.set_zlim(-plot_size, plot_size)   # z-axis limits
    ax.tick_params(axis='both', which='major', labelsize=6)
    ax.set_xlabel('x (kpc)')
    ax.set_ylabel('y (kpc)')
    ax.set_zlabel('z (kpc)')
    ax.set_title(f'{merger} merger: 3D Halo Structure t={t:.2f}')
    return fig 
    

def Plothalo_planes(x,y,z,t):
    plt.figure(figsize=(10, 6))
    plt.plot(x,y, 'o')
    plt.xlabel('x (kpc)')
    plt.ylabel('y (kpc)')
    plt.title(f'Flattened to xy plane t={t:.2f}')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(x,z, 'o')
    plt.xlabel('x (kpc)')
    plt.ylabel('z (kpc)')
    plt.title(f'Flattened to xz plane t={t:.2f}')
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(y,z, 'o')
    plt.xlabel('y (kpc)')
    plt.ylabel('z (kpc)')
    plt.title(f'Flattened to yz plane t={t:.2f}')
    plt.grid(True, alpha=0.3)
    plt.show()
    return
    
 
    