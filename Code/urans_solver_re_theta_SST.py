# ==========================================
# Block 1 Unified Imports & User Configuration (Preprocessing)
# ==========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
# --- Plotting style ---
sns.set(style=whitegrid, font_scale=1.2)

def get_reynolds_numbers()
        re_in = input(Enter the single Reynolds number value (ENTER for 250000) )
        re = float(re_in) if re_in else 250000.0
        return re
def get_aoa_values()
    aoa_in = input(Enter the single AoA value (degrees, ENTER for 15) )
    aoa = float(aoa_in) if aoa_in else 15.0
    return aoa

# --- UserProject Configuration ---
airfoil = '0018'
default_re = get_reynolds_numbers()
default_aoa = get_aoa_values()
solver_mode = 'transitional'
# --- Physical constants & domain geometry ---
rho      = 1.225
mu       = 1.7894e-5
chord    = 1.0
domain_x = (-7, 13.0)
domain_y = (-5, 5)
nx, ny     = 256, 128
mesh_label = fine
# --- Derived freestream velocities ---
u_refs = [default_re  mu  (rho  chord)]
from aerosandbox.geometry.airfoil.airfoil import Airfoil

# Tell it how many points up‐front
af = Airfoil(NACA0018, n_points=200)
# af.coordinates is now a (200×2) ndarray [[x0,y0], [x1,y1], …]
coords = af.coordinates
# Split into x & y
x_af_ur, y_af_ur = coords[, 0], coords[, 1]
θ = np.deg2rad(default_aoa)
cosθ, sinθ = np.cos(θ), np.sin(θ)
xy = np.vstack((x_af_ur, y_af_ur))              # shape (2, N)
xy_rot = np.array([[ cosθ, sinθ],
                   [-sinθ, cosθ]]) @ xy    # (2×2)·(2×N) ⇒ (2×N)
x_af, y_af = xy_rot[0,], xy_rot[1,]
# Quick plot to verify
import matplotlib.pyplot as plt
plt.figure(figsize=(7,2.5))
plt.plot(x_af, y_af, '-k', lw=1.5)
plt.axis('equal')
plt.grid(True)
plt.xlabel('x  chord')
plt.ylabel('y  chord')
plt.title('NACA 0018 from AeroSandbox')
plt.show()

# ==========================================
# Block 3 Three‐Zone Structured Mesh (Coarse Everywhere, Medium sized middle mesh, and Fine Inside Rectangle)
# ==========================================
# 1) Define the “fine” rectangle (±0.2·c around the airfoil)
dx_refine = 0.2  chord
x_min_af, x_max_af = x_af.min(), x_af.max()
y_min_af, y_max_af = y_af.min(), y_af.max()
xmin_fine = x_min_af - dx_refine
xmax_fine = x_max_af + dx_refine
ymin_fine = y_min_af - dx_refine
ymax_fine = y_max_af + dx_refine

# 2) Define the “medium” rectangle (±1·c around the airfoil)
dx_med_refine = 1.25  chord
x_min_af, x_max_af = x_af.min(), x_af.max()
y_min_af, y_max_af = y_af.min(), y_af.max()
xmin_med = x_min_af - dx_med_refine
xmax_med = x_max_af + dx_med_refine
ymin_med = y_min_af - dx_med_refine
ymax_med = y_max_af + dx_med_refine

# 2) Choose cell counts
NX_coarse, NY_coarse = 100, 50   # coarse grid divisions in xy
NX_med, NY_med = 60, 30 # medium size middle zone.
NX_fine,   NY_fine   = 200, 100   # fine grid divisions inside rectangle

# 3) Build coarse‐grid coordinates (full domain)
x_coarse = np.linspace(domain_x[0], domain_x[1], NX_coarse + 1)
y_coarse = np.linspace(domain_y[0], domain_y[1], NY_coarse + 1)
x_med = np.linspace(xmin_med, xmax_med, NX_med + 1)
y_med = np.linspace(ymin_med, ymax_med, NY_med + 1)
# 4) Build fine‐grid coordinates (just inside the rectangle)
x_fine = np.linspace(xmin_fine, xmax_fine, NX_fine + 1)
y_fine = np.linspace(ymin_fine, ymax_fine, NY_fine + 1)

# 5) Plot the mesh
plt.figure(figsize=(8,4))
# 5a) Plot coarse mesh lines (full extent)
for x0 in x_coarse plt.plot([x0, x0], [domain_y[0], domain_y[1]], color='gray', lw=0.7)
for y0 in y_coarse plt.plot([domain_x[0], domain_x[1]], [y0, y0], color='gray', lw=0.7)
for x0 in x_med plt.plot([x0, x0], [ymin_med, ymax_med], color='gray', lw=0.5)
for y0 in y_med plt.plot([xmin_med, xmax_med], [y0, y0], color='gray', lw=0.5)
# 5b) Plot fine mesh lines (only inside the rectangle)
for x0 in x_fine plt.plot([x0, x0], [ymin_fine, ymax_fine], color='gray', lw=0.3)
for y0 in y_fine plt.plot([xmin_fine, xmax_fine], [y0, y0], color='gray', lw=0.3)
# 6) Overlay the airfoil
plt.plot(x_af, y_af, 'r-', lw=1.5, label=f'NACA {airfoil}')
# 7) Decorations
plt.axis('equal')
plt.xlim(domain_x)
plt.ylim(domain_y)
plt.xlabel('x [chord]')
plt.ylabel('y [chord]')
plt.title('Three-Zone Structured Mesh Fine  Medium  Coarse')
plt.legend(loc='lower right')
plt.tight_layout()
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()
# ==========================================
# Block 4 (updated) Fill & Flag Airfoil Interior as Wall
# ==========================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
# 1) Reconstruct mesh grid coordinates
# new include medium zone
x_all = np.unique(np.concatenate([x_coarse, x_med,   x_fine ]))
y_all = np.unique(np.concatenate([y_coarse, y_med,   y_fine ]))
Xg, Yg = np.meshgrid(x_all, y_all)
pts = np.column_stack((Xg.ravel(), Yg.ravel()))
# 2) Initialize BC flags (0=interior,1=inlet,2=outlet,3=far,4=wall)
bc_flag = np.zeros(pts.shape[0], dtype=int)
# 3) Tag inlet, outlet, far‐field
tol_edge = 1e-3
bc_flag[np.abs(pts[,0] - domain_x[0])  tol_edge] = 1
bc_flag[np.abs(pts[,0] - domain_x[1])  tol_edge] = 2
bc_flag[np.abs(pts[,1] - domain_y[0])  tol_edge] = 3
bc_flag[np.abs(pts[,1] - domain_y[1])  tol_edge] = 3
# 4) Create a polygon of the airfoil and test which points lie inside
airfoil_poly = Path(np.column_stack((x_af, y_af)))
# contains_points returns True for points strictly inside the polygon
inside_mask = airfoil_poly.contains_points(pts)
# 5) Flag all interior and boundary airfoil points as wall (4)
bc_flag[inside_mask] = 4
# 6) Plotting
plt.figure(figsize=(8,4))
# plot coarse mesh lines lightly
for x0 in x_all
    plt.plot([x0, x0], [domain_y[0], domain_y[1]], color='lightgray', lw=0.5)
for y0 in y_all
    plt.plot([domain_x[0], domain_x[1]], [y0, y0], color='lightgray', lw=0.5)
# overlay BC regions
colors = {1'blue', 2'green', 3'cyan', 4'black'}
labels = {1'Inlet', 2'Outlet', 3'Far-field', 4'Wall'}
for flag in [1,2,3,4]
    pts_flag = pts[bc_flag == flag]
    if pts_flag.size
        plt.scatter(pts_flag[,0], pts_flag[,1], color=colors[flag], s=8, label=labels[flag])

# outline the airfoil
plt.plot(x_af, y_af, 'r-', lw=1.5)
plt.axis('equal')
plt.xlim(domain_x)
plt.ylim(domain_y)
plt.xlabel('x [chord]')
plt.ylabel('y [chord]')
plt.title('Boundary Condition Tags (Wall = whole airfoil interior)')
plt.legend(loc='upper right', ncol=2)
plt.tight_layout()
plt.xlim(-0.25, 1.25)
plt.ylim(-0.25, 0.25)
plt.show()
# ==========================================
# Block 5 URANS γ–Reθ–SST with θ-Scheme (θ=0.5) for All Equations,
#          Upwind Advection, Trapezoidal Time Integration,
#          Snapshots, SST-notify and Solver-Mode Prints
# ==========================================
import os
import numpy as np
from scipy.sparse import diags, identity
from scipy.sparse.linalg import spsolve
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata  # -- ADD THIS LINE

def laplacian_matrix(nx, ny, dx, dy)
    N = nx  ny
    main  = -2(1dx2 + 1dy2)  np.ones(N)
    offx  =     1dx2      np.ones(N-1)
    offx[np.arange(1, N) % nx == 0] = 0
    offy  =     1dy2      np.ones(N-nx)
    A = diags([main, offx, offx, offy, offy],
              [0,   -1,    1,   -nx,   nx],
              shape=(N, N)).tocsr()
    A[0,] = 0; A[0,0] = 1
    return A

def build_upwind_advection_matrix(nx, ny, dx, dy, ux, uy)
    N = nx  ny
    ux_flat = ux.ravel(); uy_flat = uy.ravel()
    pos_x = np.maximum(ux_flat, 0)dx;  neg_x = np.minimum(ux_flat, 0)dx
    pos_y = np.maximum(uy_flat, 0)dy;  neg_y = np.minimum(uy_flat, 0)dy
    main = -(pos_x - neg_x + pos_y - neg_y)
    A = diags([main, pos_x, -neg_x, pos_y, -neg_y],
              [0,   -1,     1,   -nx,    nx],
              shape=(N,N)).tocsr()
    return A

def poisson_2d(rhs, dx, dy)
    ny, nx = rhs.shape
    A = laplacian_matrix(nx, ny, dx, dy)
    b = rhs.ravel(); b[0] = 0
    return spsolve(A, b).reshape(ny, nx)

def apply_bc_field(phi, bc, val_if, val_wall)
    phi[(bc==1) (bc==3)] = val_if
    phi[ bc==4 ] = val_wall
    return phi

def enforce_bc_matrix(M, rhs, bc, nx, ny, val_if, val_wall)
    M = M.tolil()
    for j in range(ny)
        for i in range(nx)
            idx = jnx + i
            f   = bc[j,i]
            if f in (1,3)      # inlet or far
                M.rows[idx] = [idx]; M.data[idx] = [1.0]
                rhs[idx]    = val_if
            elif f == 4        # wall
                M.rows[idx] = [idx]; M.data[idx] = [1.0]
                rhs[idx]    = val_wall
    return M.tocsr(), rhs

def run_fully_implicit(dt,n_steps,snapshot_interval=100,print_every=30,
        sst_notify_interval=300,solver_print_interval=300,θ=0.5,relax=0.4)
    # rebuild mesh & BC
    # new include medium zone
    x_all = np.unique(np.concatenate([x_coarse, x_med,   x_fine ]))
    y_all = np.unique(np.concatenate([y_coarse, y_med,   y_fine ]))
    Xg, Yg = np.meshgrid(x_all, y_all)
    ny, nx = Xg.shape
    bc     = bc_flag.reshape(ny, nx)
    dx     = (domain_x[1]-domain_x[0])(nx-1)
    dy     = (domain_y[1]-domain_y[0])(ny-1)
    # freestream
    u_inf = default_re  mu  (rho  chord)
    aoa   = np.deg2rad(default_aoa)
    u0, v0 = u_inf, 0
    # initialize fields
    gamma0    = np.zeros((ny,nx));
    gamma0[bc==1] = 1e-6
    gamma0 = apply_bc_field(gamma0, bc, val_if=1e-6, val_wall=0.0)
    k0        = np.full((ny,nx), 1e-8)
    k0 = apply_bc_field(k0, bc, val_if=1e-5, val_wall=1e-8)
    w0        = np.full((ny,nx), 1e-2)
    w0 = apply_bc_field(w0, bc, val_if=1e-2, val_wall=5.0)
    u        = np.zeros((ny,nx)); v        = np.zeros((ny,nx))
    u[,] = u0
    v[,] = v0
    p        = np.zeros((ny,nx))
    k        = k0.copy()
    omega    = w0.copy()
    gamma    = gamma0.copy()
    Re_theta = np.full((ny,nx), 700.0)
    #Re_theta[bc==1] = 2000.0  # or even 1500
    Re_theta = apply_bc_field(Re_theta, bc, val_if=10000.0, val_wall=100.0)
    gamma   = gamma0.copy()
    k       = k0.copy()
    omega   = w0.copy()
    # operators & constants
    L = laplacian_matrix(nx, ny, dx, dy)
    I = identity(nxny, format='csr')
    σk, σω, βstar = 0.85, 0.5, 0.09
    α_sst, β_sst  = 59, 340
    Cγ, Dγ, Reθ_c     = 50.0, 3.0, 350.0
    Prθ, Prγ      = 0.9, 0.9
    νt_max, ω_min, k_min, k_max = 5000.0, 1e-3, 1e-8, 0.5  u_inf2
    Vmag_snapshots = []; u_snapshots = []; v_snapshots = []; times = []; CL_history = []; CD_history = []; time_history =[]
    output_dir = snapshots
    os.makedirs(output_dir, exist_ok=True)
    # make sure x_af, y_af define a closed loop (x_af[-1]==x_af[0])
    dx_panels = x_af[1] - x_af[-1]
    dy_panels = y_af[1] - y_af[-1]
    L_panels  = np.hypot(dx_panels, dy_panels)
    area_sign = np.sign(np.sum(x_af[-1]y_af[1] - x_af[1]y_af[-1]))
    nxnew =  area_sign  (dy_panels  L_panels)
    nynew = -area_sign  (dx_panels  L_panels)
    # (Optionally) mid‐point locations on each panel
    x_mid = 0.5(x_af[-1] + x_af[1])
    y_mid = 0.5(y_af[-1] + y_af[1])

    for step in tqdm(range(1, n_steps+1), desc=URANS θ-scheme)
        # store old
        u_old, v_old   = u.copy(), v.copy()
        k_old, ω_old   = k.copy(), omega.copy()
        γ_old, Reθ_old = gamma.copy(), Re_theta.copy()
        # turbulent viscosity
        nu_t      = np.clip(γ_old(k_oldomega), 0.0, νt_max)
        nu_t_flat = nu_t.ravel()
        # assemble momentum operator
        A_adv  = build_upwind_advection_matrix(nx, ny, dx, dy, u_old, v_old)
        D_diff = diags(nu_t_flat,0) @ L
        A_op   = A_adv + D_diff
        # —————————————————————————————————————————————
        # 1) momentum (trapezoidal θ-scheme)
        M_lhs = I - θdtA_op
        M_rhs = I + (1-θ)dtA_op
        u_rhs = M_rhs @ u_old.ravel()
        v_rhs = M_rhs @ v_old.ravel()
        # pressure gradient treated explicitly
        dpdx_old = np.gradient(p, dx, axis=1).ravel()
        dpdy_old = np.gradient(p, dy, axis=0).ravel()
        u_rhs -= dt(1rho)dpdx_old
        v_rhs -= dt(1rho)dpdy_old
        M_u, u_rhs = enforce_bc_matrix(M_lhs.copy(), u_rhs, bc, nx, ny, u0, 0.0)
        M_v, v_rhs = enforce_bc_matrix(M_lhs.copy(), v_rhs, bc, nx, ny, v0, 0.0)
        u_star = spsolve(M_u, u_rhs).reshape(ny,nx)
        v_star = spsolve(M_v, v_rhs).reshape(ny,nx)
        # pressure‐projection (unchanged)
        div_star = ((u_star[1-1,2]-u_star[1-1,-2])(2dx)
                  + (v_star[2,1-1]-v_star[-2,1-1])(2dy))
        rhs_p    = np.zeros_like(p)
        rhs_p[1-1,1-1] = rhodt  div_star
        p = poisson_2d(rhs_p, dx, dy)
        dpdx = np.gradient(p, dx, axis=1)
        dpdy = np.gradient(p, dy, axis=0)
        u_new = u_star - dt(1rho)dpdx
        v_new = v_star - dt(1rho)dpdy
        # re-apply velocity BCs
        u_new = apply_bc_field(u_new, bc, u0, 0.0)
        v_new = apply_bc_field(v_new, bc, v0, 0.0)
        u = relaxu_new + (1-relax)u_old
        v = relaxv_new + (1-relax)v_old
        # —————————————————————————————————————————————
        # 2) k-equation (θ-scheme)
        dudx = np.gradient(u, dx, axis=1)
        dvdy = np.gradient(v, dy, axis=0)
        S2   = 2(dudx2 + dvdy2)
        Pk   = np.clip(γ_old(k_oldomega)S2, 0.0, 1e5)
        K_op = A_adv + diags((murho + nu_t_flatσk),0)@L 
               + diags((βstaromega).ravel(),0)
        M_k_lhs = I - θdtK_op
        M_k_rhs = (I + (1-θ)dtK_op) @ k_old.ravel() + dtPk.ravel()
        M_k, rhs_k = enforce_bc_matrix(M_k_lhs, M_k_rhs, bc, nx, ny, 1e-5, 1e-8)
        k_new = spsolve(M_k, rhs_k).reshape(ny,nx)
        k     = relaxnp.clip(k_new, k_min, k_max) + (1-relax)k_old
        # —————————————————————————————————————————————
        # 3) ω-equation (θ-scheme)
        ufac  = (murho) + nu_t_flat
        W_op  = A_adv + diags((murho + nu_t_flatσω),0)@L 
               + diags((β_sstomega).ravel(),0)
        ω_src = (α_sstPk.ravel()np.maximum(ufac,1e-8))
        M_w_lhs = I - θdtW_op
        M_w_rhs = (I + (1-θ)dtW_op) @ omega.ravel() + dtω_src
        M_w, rhs_w = enforce_bc_matrix(M_w_lhs, M_w_rhs, bc, nx, ny, 1e-3, 5.0)
        ω_new = spsolve(M_w, rhs_w).reshape(ny,nx)
        omega = relaxnp.clip(ω_new, ω_min, 20000) + (1-relax)ω_old
        # —————————————————————————————————————————————
        # 4) Reθ‐equation (θ‐scheme)
        # flatten both pieces so they broadcast correctly
        term1 = (np.sqrt(uu + vv)  (rho  Prθ)).ravel()
        term2 = ((Re_theta - np.roll(Re_theta, 1, axis=1))  dx).ravel()
        Pth   = term1  term2
        # build operators & RHS
        Th_op    = A_adv + LPrθ
        th_src   = Pth - 0.0# 1  (Reθ_old.ravel() - 100.0)
        M_th_lhs = I - θ  dt  Th_op
        M_th_rhs = (I + (1-θ)  dt  Th_op) @ Reθ_old.ravel() + dt  th_src
        M_th, rhs_th = enforce_bc_matrix( M_th_lhs, M_th_rhs, bc, 
            nx, ny, val_if=2000.0,val_wall=100.0)
        th_new     = spsolve(M_th, rhs_th).reshape(ny, nx)
        Re_theta  = relax  np.clip(th_new, 50.0, 1e5) + (1-relax)Reθ_old
        # —————————————————————————————————————————————
        # 5) γ-equation (θ-scheme)
        # flatten intermittency and build productiondestruction terms
        F_len_flat  = np.clip(Re_thetaReθ_c, 0.0, 1.0).ravel()
        Pg_flat     = Cγ  F_len_flat  (1 - γ_old.ravel())
        Dg_flat     = Dγ  γ_old.ravel()

        # build trapezoidal (θ‐scheme) operators
        G_op        = A_adv + LPrγ
        M_g_lhs     = I - θ  dt  G_op
        M_g_rhs     = (I + (1 - θ)  dt  G_op) @ γ_old.ravel() + dt  (Pg_flat - Dg_flat)

        # enforce BCs and solve
        M_g, rhs_g  = enforce_bc_matrix(
            M_g_lhs, M_g_rhs, bc, nx, ny,
            val_if=0.0,val_wall=0.0)
        gamma_new_flat = spsolve(M_g, rhs_g)
        gamma_new      = gamma_new_flat.reshape(ny, nx)
        gamma_new      = np.clip(gamma_new, 0.0, 1.0)
        gamma          = apply_bc_field(gamma_new, bc, val_if=0.0, val_wall=0.0)
        # prints
        if step % print_every == 0 or step =5
            dγ_dt   = (gamma - γ_old)  dt
            dk_dt   = (k - k_old)  dt
            dω_dt   = (omega - ω_old)  dt
            dReθ_dt = (Re_theta - Reθ_old)  dt
            print(f[Step {step}{n_steps}])
            print(f  u     min={u.min()8.3e}, max={u.max()8.3e})
            print(f  v     min={v.min()8.3e}, max={v.max()8.3e})
            print(f  p     min={p.min()8.3e}, max={p.max()8.3e})
            print(f  k     min={k.min()8.3e}, max={k.max()8.3e})
            print(f  ω     min={omega.min()8.3e}, max={omega.max()8.3e})
            print(f  γ     min={gamma.min()8.3e}, max={gamma.max()8.3e})
            print(f  Reθ   min={Re_theta.min()8.3e}, max={Re_theta.max()8.3e})
            print(f  νt    min={nu_t.min()8.3e}, max={nu_t.max()8.3e})
            print(f  Pk    min={Pk.min()8.3e}, max={Pk.max()8.3e})
            print(f  Pg    min={Pg_flat.min()8.3e}, max={Pg_flat.max()8.3e})
            print(f  Dg    min={Dg_flat.min()8.3e}, max={Dg_flat.max()8.3e})
            print(f  ∂γ∂t min={dγ_dt.min()8.3e}, max={dγ_dt.max()8.3e})
            print(f  ∂k∂t min={dk_dt.min()8.3e}, max={dk_dt.max()8.3e})
            print(f  ∂ω∂t min={dω_dt.min()8.3e}, max={dω_dt.max()8.3e})
            print(f  ∂Reθ∂t min={dReθ_dt.min()8.3e}, max={dReθ_dt.max()8.3e})
            print(f  A_adv_∞ = {abs(A_adv).max().3e}, D_diff_∞ = {abs(D_diff).max().3e})
            print(-  60)
            # interpolate the current p field onto the surface mid-points
            p_mid = griddata((Xg.ravel(), Yg.ravel()),p.ravel(),
                (x_mid, y_mid),method='linear')
            # integrate pressure to get Fx, Fy
            Fx = -np.sum(p_mid  L_panels  nxnew)
            Fy = -np.sum(p_mid  L_panels  nynew)
            # project into lift & drag, then nondimensionalize
            α     = np.deg2rad(default_aoa)
            Lift  = -Fxnp.sin(α) + Fynp.cos(α)
            Drag  =  Fxnp.cos(α) + Fynp.sin(α)
            q_inf = 0.5  rho  u_inf2
            CL = Lift  (q_inf  chord)
            CD = Drag  (q_inf  chord)
            # store or print
            print(f  CL = {CL.4f}, CD = {CD.4f})
            CL_history.append(CL)
            CD_history.append(CD)
            time_history.append(step  dt)

        if step % 50 == 0 or step == 3
            plt.figure(figsize=(7, 3.5))
            # Velocity magnitude field
            Vmag = np.sqrt(u2 + v2)
            cf = plt.contourf(Xg, Yg, Vmag, levels=30, cmap='viridis')
            plt.colorbar(cf, label='V [ms]')
            # Airfoil outline
            # Interpolated velocity field for streamlines
            xi = np.linspace(-0.5,1.5)  # zoom in around the airfoil
            yi = np.linspace(-0.5, 0.5)
            XI, YI = np.meshgrid(xi, yi)
            U = griddata((Xg.ravel(), Yg.ravel()), u.ravel(), (XI, YI), method='linear')
            V = griddata((Xg.ravel(), Yg.ravel()), v.ravel(), (XI, YI), method='linear')
            plt.streamplot(XI, YI, U, V, color='yellow', density=1.0, linewidth=0.9)
            plt.fill(x_af, y_af, color='midnightblue', zorder=4)      # hide any streamlines behind
            plt.plot(x_af, y_af, 'k-', lw=2)
            plt.title(f'NACA {airfoil}  Re={default_re.0f}, AoA={default_aoa.1f}°, Step {step}   t = {stepdt.3f} s')
            #plt.title(fStep {step}   t = {stepdt.3f} s)
            plt.xlabel('x  chord')
            plt.ylabel('y  chord')
            plt.axis('equal')
            plt.xlim(-0.5, 1.5)  # Focused zoom
            plt.ylim(-0.5, 0.5)
            plt.tight_layout()
            # build your figure exactly as before…
            # now save it
            filename = os.path.join(output_dir, fVmag_step_{step04d}.png)
            plt.savefig(filename, dpi=300)     # dpi=300 gives publication-quality
            plt.pause(0.01)
            plt.close()
            p_inf  = 0.0      # assuming your p starts at zero freestream gauge
            q_inf  = 0.5  rho  u_inf2
            cp_field = (p - p_inf)  q_inf
            # 2) Filled contour of C_p
            plt.figure(figsize=(7,3.5))
            levels = np.linspace(-3, 1, 50)   # adjust depending on peak suction
            cf = plt.contourf(
                Xg, Yg, cp_field,
                levels=levels,
                cmap=viridis_r,   # reversed viridis dark = suction
                extend=both
            )
            plt.colorbar(cf, label='$C_p$')
            # 3) Airfoil & streamlines on top (just copy your streamline code)
            plt.streamplot(XI, YI, U, V,
                color='white', density=1.0, linewidth=0.7)
            plt.fill(x_af, y_af, color='k', zorder=4)
            plt.plot(x_af, y_af, 'k-', lw=2)
            # 4) Labels & limits
            plt.title(f'Pressure-Coefficient Field, Step {step}, t={stepdt.3f}s')
            plt.xlabel('x  chord')
            plt.ylabel('y  chord')
            plt.axis('equal')
            plt.xlim(-0.5,1.5)
            plt.ylim(-0.5,0.5)
            # 5) Save or show
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, fCP_step_{step04d}.png), dpi=300)
            plt.close()
        if np.isnan(gamma).any() or np.isnan(k).any() or np.isnan(u).any()
            print(f[Step {step}] NaN detected! Halting early.)
            break
        if step % solver_print_interval == 0
            mode = TURBULENT if gamma.max()=0.5 else LAMINAR
            print(f[Step {step}] Solver mode {mode})
        # snapshots
        if snapshot_interval and (step % snapshot_interval == 0)
            Vmag = np.sqrt(u2 + v2)
            Vmag_snapshots.append(Vmag.copy())
            u_snapshots.append(u.copy())
            v_snapshots.append(v.copy())
            times.append(stepdt)
    return u, v, p, k, omega, gamma, Re_theta, Vmag_snapshots, u_snapshots, v_snapshots, times, CL_history, CD_history, time_history
# --- Solver Call ---
dt      = 5e-3
n_steps = 1800
u, v, p, k, omega, gamma, Re_theta, Vmag_snapshots, u_snapshots, v_snapshots, times, CL_history, CD_history, time_history = 
    run_fully_implicit(dt, n_steps, snapshot_interval=30,print_every=30,solver_print_interval=50)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.interpolate import griddata
from matplotlib.path import Path
from matplotlib import cm

# 1) Reconstruct mesh & airfoil mask
x_all        = np.unique(np.concatenate([x_coarse, x_med, x_fine]))
y_all        = np.unique(np.concatenate([y_coarse, y_med, y_fine]))
Xg, Yg       = np.meshgrid(x_all, y_all)
airfoil_poly = Path(np.column_stack((x_af, y_af)))
# 1a) Split foil into upper & lower for distance interp
coords = np.column_stack((x_af, y_af))
upper  = coords[coords[,1] = 0]
lower  = coords[coords[,1] = 0]
upper  = upper[np.argsort(upper[,0])]
lower  = lower[np.argsort(lower[,0])]
xu, yu = upper.T
xl, yl = lower.T
# 2) Uniform plotting grid for streamlines
xi, yi = np.linspace(domain_x[0], domain_x[1], 300), np.linspace(domain_y[0], domain_y[1], 150)
XI, YI = np.meshgrid(xi, yi)
fig, ax = plt.subplots(figsize=(8,4), constrained_layout=True)
cmap = cm.viridis
# Pre-compute global Vmag range for a single colorbar
Vmin = min(V.min() for V in Vmag_snapshots)
Vmax = max(V.max() for V in Vmag_snapshots)
norm = plt.Normalize(Vmin, Vmax)
sm   = cm.ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
fig.colorbar(sm, ax=ax, label=V [ms])
dist_tol = 0.05  # capture bubble within ±0.05 c of the surface
def update(frame)
    ax.clear()
    # — (a) background speed‐contour —
    V = Vmag_snapshots[frame]
    cf = ax.contourf(Xg, Yg, V,
        levels=50, cmap=cmap, norm=norm )
    # — (b) streamlines —
    U = griddata((Xg.ravel(), Yg.ravel()),
                 u_snapshots[frame].ravel(),
                 (XI, YI), method=linear)
    W = griddata((Xg.ravel(), Yg.ravel()),
                 v_snapshots[frame].ravel(),
                 (XI, YI), method=linear)
    mask = airfoil_poly.contains_points(np.column_stack((XI.ravel(), YI.ravel())))
    Um, Wm = (np.ma.array(U, mask=mask.reshape(XI.shape)),
              np.ma.array(W, mask=mask.reshape(XI.shape)))
    ax.streamplot(XI, YI, Um, Wm, color=yellow, density=4, linewidth=1)
    # — (c) foil outline —
    ax.plot(x_af, y_af, 'k-', lw=2, zorder=50)
    # — (d) separation bubble via distance‐to‐foil mask —
    XIf = XI.ravel(); YIf = YI.ravel()
    # interpolate local foil y at each XI
    yui = np.interp(XIf, xu, yu)
    yli = np.interp(XIf, xl, yl)
    # vertical distance to foil
    d_up   = np.abs(YIf - yui)
    d_down = np.abs(YIf - yli)
    dist2foil = np.minimum(d_up, d_down).reshape(XI.shape)
    # recirculation & near foil
    #sep_mask = (U  3) & (dist2foil  dist_tol)
    #ax.contour(        XI, YI, sep_mask.astype(float),levels=[0.5],        colors='orange',        linewidths=2,        linestyles='--',        zorder=60)
    # — decorations —
    ax.set_title(fNACA {airfoil}  Re={default_re.0f}, AoA={default_aoa.1f}°  t = {times[frame].3f} s)
    ax.set_aspect(equal,box)
    ax.set_xlim(-1.5, 2.5)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(x  chord)
    ax.set_ylabel(y  chord)
# build & save animation
ani = FuncAnimation(fig, update, frames=len(Vmag_snapshots), interval=100, blit=False)
ani.save(wake_with_separation.gif, writer=PillowWriter(fps=10))
plt.close(fig)
from IPython.display import Image, display
display(Image(filename=wake_with_separation.gif))
plt.figure(figsize=(10, 4))
plt.plot(time_history, CD_history, label=$C_D$, lw=2)
plt.xlabel(Time [s])
plt.ylabel(CD)
plt.xlim(0,0.5)
plt.title(fDrag Coefficient vs Time  NACA {airfoil}  Re={default_re.0f}, AoA={default_aoa.1f}°)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(cd_vs_time_plot.png, dpi=300)
plt.show()
plt.figure(figsize=(10, 4))
plt.plot(time_history, CD_history, label=$C_L$, lw=2)
plt.xlabel(Time [s])
plt.ylabel(CL)
plt.xlim(0,0.5)
plt.title(fLift Coefficient vs Time  NACA {airfoil}  Re={default_re.0f}, AoA={default_aoa.1f}°)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(cl_vs_time_plot.png, dpi=300)
plt.show()
plt.figure(figsize=(10, 4))
plt.plot(time_history, (np.array(CL_history)np.array(CD_history)), label=$C_LC_D$, lw=2)
plt.xlabel(Time [s])
plt.ylabel(CLCD)
plt.xlim(0,1)
plt.title(fLiftDrag Coefficients vs Time  NACA {airfoil}  Re={default_re.0f}, AoA={default_aoa.1f}°)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(clcd_vs_time_plot.png, dpi=300)
plt.show()