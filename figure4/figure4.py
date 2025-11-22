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
# parameters & sweep setup
# -------------------------
p_base = [5.0, 0.1, 2.0, 2.0, 0.4, 0.01]   # a1,a2,b1,b2,d1,d2 (we'll overwrite b1)
u0 = [0.76, 0.16, 9.9]
t0, tf = 0.0, 10000.0

bs = np.arange(2.0, 6.2000001, 0.01)   # inclusive of 6.2
N = 1000                                # number of last z values to collect
t_points = 20001                        # number of time points (≈ same resolution as earlier translations)
t_eval = np.linspace(t0, tf, t_points)

nb = len(bs)
# storage: Z shape (N, nb)
Z = np.zeros((N, nb), dtype=float)
Zmax = np.zeros(nb, dtype=float)
Zmin = np.zeros(nb, dtype=float)
errors_idx = []

# -------------------------
# loop over b1 values
# -------------------------
for j, b1 in enumerate(bs):
    p = p_base.copy()
    p[2] = b1   # set b1

    sol = solve_ivp(lambda t, y: parameterized_model(t, y, p),
                    (t0, tf), u0, t_eval=t_eval, rtol=1e-9, atol=1e-12)

    if not sol.success:
        errors_idx.append(j)
        continue

    z_full = sol.y[2, :]   # entire z time series
    if z_full.size >= N:
        Z[:, j] = z_full[-N:]
        Zmax[j] = Z[:, j].max()
        Zmin[j] = Z[:, j].min()
    else:
        errors_idx.append(j)

# report any errors (indices into bs)
if errors_idx:
    print("Warning: solver failed or produced too-short output for these bs indices (0-based):", errors_idx)

# -------------------------
# find local maxima above threshold
# -------------------------
Zmaxloc = np.zeros_like(Z)

# iterate interior indices 1..N-2 as in Julia (i = 2:(N-1) -> python 1..N-2)
for j in range(nb):
    zcol = Z[:, j]
    zrange = Zmax[j] - Zmin[j]
    thresh = Zmin[j] + 0.66 * zrange
    # interior indices
    for i in range(1, N-1):
        if zcol[i] > zcol[i-1] and zcol[i] > zcol[i+1]:
            if zcol[i] > thresh:
                Zmaxloc[i, j] = zcol[i]

# -------------------------
# Build scatter points (b, z_maxloc) removing zeros
# -------------------------
B_repeated = np.repeat(bs, N)           # bs repeated inner = N
Zflat = Zmaxloc.flatten(order='F')      # flatten in column-major to match Julia vec(Zmaxloc)
Bref = np.repeat(bs, N)                 # same as B_repeated

# But simpler: iterate and collect nonzeros
pts = []
for j in range(nb):
    bval = bs[j]
    for i in range(N):
        zval = Zmaxloc[i, j]
        if zval > 0.0:
            pts.append((bval, zval))

if len(pts) == 0:
    raise RuntimeError("No maxima found. Try reducing threshold or increasing integration resolution.")

pts = np.array(pts)   # shape (M, 2)

# -------------------------
# Plotting: three panels stacked like Julia's fig4a,b,c
# -------------------------
fig, axes = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=False)

# common scatter kwargs
scatter_kwargs = dict(s=2, alpha=0.5, color='black', linewidths=0)

# Fig 4A: xlim (2.2, 3.2), ylim (9.5, 13.0)
ax = axes[0]
ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
ax.set_xlim(2.2, 3.2)
ax.set_ylim(9.5, 13.0)
ax.set_xlabel(r"$b_1$")
ax.set_ylabel(r"$z_{\max}$")
ax.set_title("a", loc='left')
ax.grid(False)

# Fig 4B: xlim (3.0, 6.5), ylim (3.0, 10.0), custom ticks
ax = axes[1]
ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
ax.set_xlim(3.0, 6.5)
ax.set_ylim(3.0, 10.0)
ax.set_xticks(np.arange(3.0, 6.6, 0.5))
ax.set_yticks(np.arange(3.0, 11.0, 1.0))
ax.set_xlabel(r"$b_1$")
ax.set_ylabel(r"$z_{\max}$")
ax.set_title("b", loc='left')
ax.grid(False)

# Fig 4C: xlim (2.25, 2.6), ylim (11.4, 12.8), fine ticks
ax = axes[2]
ax.scatter(pts[:, 0], pts[:, 1], **scatter_kwargs)
ax.set_xlim(2.25, 2.6)
ax.set_ylim(11.4, 12.8)
ax.set_xticks(np.arange(2.25, 2.61, 0.05))
ax.set_yticks(np.arange(11.1, 12.9, 0.2))
ax.set_xlabel(r"$b_1$")
ax.set_ylabel(r"$z_{\max}$")
ax.set_title("c", loc='left')
ax.grid(False)

# global formatting (tick sizes, margins)
for ax in axes:
    ax.tick_params(labelsize=8, direction='out')

plt.subplots_adjust(hspace=0.25, left=0.12, right=0.95, top=0.98, bottom=0.02)

# ensure dir and save
out_dir = os.path.join("..", "article", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig4.pdf")
fig.savefig(out_path, bbox_inches='tight')
print("Saved bifurcation figure to:", out_path)
