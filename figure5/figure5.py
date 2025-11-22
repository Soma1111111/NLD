import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# -------------------------
# model (same algebra)
# -------------------------
def f(u, a, b):
    return a * u / (1.0 + b * u)

def parameterized_model(t, u, p):
    x, y, z = u
    a1, a2, b1, b2, d1, d2 = p
    dx = x * (1.0 - x) - f(x, a1, b1) * y
    dy = f(x, a1, b1) * y - f(y, a2, b2) * z - d1 * y
    dz = f(y, a2, b2) * z - d2 * z
    return [dx, dy, dz]

# -------------------------
# common settings
# -------------------------
u0 = [0.76, 0.16, 9.9]
t0, tf = 0.0, 10000.0
# dense time grid so indexing behavior resembles Julia's sol[i]
t_points = 20001
t_eval = np.linspace(t0, tf, t_points)
out_dir = os.path.join("..", "article", "figures")
os.makedirs(out_dir, exist_ok=True)

# small helper to compute concordant indices and filter consecutive hits
def compute_concordant_indices(sol_x, sol_y, sol_z, cond_x, cond_y, cond_z):
    """
    cond_x, cond_y, cond_z are callables that accept an array and return boolean mask.
    Returns sorted, filtered indices where all three conditions true and consecutive duplicates removed
    by dropping the earlier index of any consecutive pair (matches Julia code behavior).
    """
    mask_x = cond_x(sol_x)
    mask_y = cond_y(sol_y)
    mask_z = cond_z(sol_z)
    concordant_mask = mask_x & mask_y & mask_z
    indices = np.nonzero(concordant_mask)[0]  # 0-based indices
    if indices.size == 0:
        return np.array([], dtype=int)
    # drop earlier index when two indices are consecutive:
    filtered = []
    for k, idx in enumerate(indices):
        if k == 0:
            filtered.append(idx)
        else:
            if indices[k-1] == idx - 1:
                # drop the earlier one -> replace last appended with current
                # Julia set earlier index to 0 then removed zeros; net effect: keep later index
                filtered[-1] = idx
            else:
                filtered.append(idx)
    return np.array(filtered, dtype=int)

# -------------------------
# PART A/B: b1 = 3.0
# -------------------------
p_b3 = [5.0, 0.1, 3.0, 2.0, 0.4, 0.01]
sol_b3 = solve_ivp(lambda t, y: parameterized_model(t, y, p_b3),
                   (t0, tf), u0, t_eval=t_eval, rtol=1e-12, atol=1e-14)
if not sol_b3.success:
    raise RuntimeError("Solver failed for b1=3.0: " + sol_b3.message)

sol_x_b3 = sol_b3.y[0, :]
sol_y_b3 = sol_b3.y[1, :]
sol_z_b3 = sol_b3.y[2, :]

constant_b3 = 9.0
eps_b3 = 0.05

# conditions (mimic Julia ranges exactly)
cond_x_b3 = lambda arr: (arr >= 0.9) & (arr <= 1.0)
cond_y_b3 = lambda arr: (arr >= 0.0) & (arr <= 0.1)
cond_z_b3 = lambda arr: (arr >= (constant_b3 - eps_b3)) & (arr <= (constant_b3 + eps_b3))

ind_xyz_b3 = compute_concordant_indices(sol_x_b3, sol_y_b3, sol_z_b3,
                                        cond_x_b3, cond_y_b3, cond_z_b3)

# Poincaré section (Fig 5A) using the filtered indices
fig5a_x = sol_x_b3[ind_xyz_b3]
fig5a_y = sol_y_b3[ind_xyz_b3]

# Poincaré map (Fig 5B): pairs x(n), x(n+1)
if ind_xyz_b3.size >= 2:
    ind_n_b3 = ind_xyz_b3[:-1]
    ind_n1_b3 = ind_xyz_b3[1:]
    fig5b_xn = sol_x_b3[ind_n_b3]
    fig5b_xn1 = sol_x_b3[ind_n1_b3]
else:
    fig5b_xn = np.array([])
    fig5b_xn1 = np.array([])

# -------------------------
# PART C/D: b1 = 6.0
# -------------------------
p_b6 = [5.0, 0.1, 6.0, 2.0, 0.4, 0.01]
sol_b6 = solve_ivp(lambda t, y: parameterized_model(t, y, p_b6),
                   (t0, tf), u0, t_eval=t_eval, rtol=1e-12, atol=1e-14)
if not sol_b6.success:
    raise RuntimeError("Solver failed for b1=6.0: " + sol_b6.message)

sol_x_b6 = sol_b6.y[0, :]
sol_y_b6 = sol_b6.y[1, :]
sol_z_b6 = sol_b6.y[2, :]

constant_b6 = 3.0
eps_b6 = 0.05

cond_x_b6 = lambda arr: (arr >= 0.93) & (arr <= 1.0)
cond_y_b6 = lambda arr: (arr >= 0.0) & (arr <= 0.085)
cond_z_b6 = lambda arr: (arr >= (constant_b6 - eps_b6)) & (arr <= (constant_b6 + eps_b6))

ind_xyz_b6 = compute_concordant_indices(sol_x_b6, sol_y_b6, sol_z_b6,
                                        cond_x_b6, cond_y_b6, cond_z_b6)

fig5c_x = sol_x_b6[ind_xyz_b6]
fig5c_y = sol_y_b6[ind_xyz_b6]

if ind_xyz_b6.size >= 2:
    ind_n_b6 = ind_xyz_b6[:-1]
    ind_n1_b6 = ind_xyz_b6[1:]
    fig5d_xn = sol_x_b6[ind_n_b6]
    fig5d_xn1 = sol_x_b6[ind_n1_b6]
else:
    fig5d_xn = np.array([])
    fig5d_xn1 = np.array([])

# -------------------------
# Plotting: 2x2 panels (A,B top row; C,D bottom row)
# -------------------------
fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=False)
(axA, axB), (axC, axD) = axs

# Fig 5A Poincaré section (b1=3.0)
xmin, xmax, ymin, ymax = 0.95, 0.983, 0.015, 0.04
axA.scatter(fig5a_x, fig5a_y, s=9, c='black')
axA.set_xlim(xmin, xmax)
axA.set_ylim(ymin, ymax)
axA.set_xticks(np.arange(xmin, xmax + 1e-6, 0.01))
axA.set_yticks(np.arange(ymin, ymax + 1e-6, 0.01))
axA.set_xlabel("x(n)")
axA.set_ylabel("y(n)")
axA.set_title("A", loc='left')
axA.grid(False)

# Fig 5B Poincaré map (b1=3.0)
xmin, xmax = 0.95, 0.98
axB.scatter(fig5b_xn, fig5b_xn1, s=9, c='black')
axB.set_xlim(xmin, xmax)
axB.set_ylim(xmin, xmax)
axB.set_xticks(np.arange(xmin, xmax + 1e-6, 0.01))
axB.set_yticks(np.arange(xmin, xmax + 1e-6, 0.01))
axB.set_xlabel("x(n)")
axB.set_ylabel("x(n+1)")
axB.set_title("B", loc='left')
# diagonal y=x
axB.plot(np.arange(xmin, xmax + 1e-6, 0.01), np.arange(xmin, xmax + 1e-6, 0.01), color='black')
axB.grid(False)

# Fig 5C Poincaré section (b1=6.0)
xmin, xmax, ymin, ymax = 0.93, 1.003, -0.003, 0.09
axC.scatter(fig5c_x, fig5c_y, s=9, c='black')
axC.set_xlim(xmin, xmax)
axC.set_ylim(ymin, ymax)
axC.set_xticks(np.arange(xmin, xmax + 1e-6, 0.02))
axC.set_yticks(np.arange(0.0, ymax + 1e-6, 0.04))
axC.set_xlabel("x(n)")
axC.set_ylabel("y(n)")
axC.set_title("C", loc='left')
axC.grid(False)

# Fig 5D Poincaré map (b1=6.0)
xmin, xmax = 0.93, 1.003
axD.scatter(fig5d_xn, fig5d_xn1, s=9, c='black')
axD.set_xlim(xmin, xmax)
axD.set_ylim(xmin, xmax)
axD.set_xticks(np.arange(xmin, xmax + 1e-6, 0.02))
axD.set_yticks(np.arange(xmin, xmax + 1e-6, 0.02))
axD.set_xlabel("x(n)")
axD.set_ylabel("x(n+1)")
axD.set_title("D", loc='left')
axD.plot(np.arange(xmin, xmax + 1e-6, 0.01), np.arange(xmin, xmax + 1e-6, 0.01), color='black')
axD.grid(False)

# global formatting
for ax in [axA, axB, axC, axD]:
    ax.tick_params(labelsize=8, direction='out')
plt.subplots_adjust(hspace=0.30, wspace=0.25, left=0.08, right=0.98, top=0.98, bottom=0.04)

out_path = os.path.join(out_dir, "fig5.pdf")
fig.savefig(out_path, bbox_inches='tight')
print("Saved fig5 to:", out_path)
