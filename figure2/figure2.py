import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# f(u, a, b) = a*u/(1 + b*u)
def f(u, a, b):
    return a * u / (1.0 + b * u)

# ODE system (matches your Julia parameterized_model!)
def parameterized_model(t, u, p):
    x, y, z = u
    a1, a2, b1, b2, d1, d2 = p
    dx = x * (1.0 - x) - f(x, a1, b1) * y
    dy = f(x, a1, b1) * y - f(y, a2, b2) * z - d1 * y
    dz = f(y, a2, b2) * z - d2 * z
    return [dx, dy, dz]

# Initial conditions, parameters, timespan
u0 = [0.76, 0.16, 9.9]
t0, tf = 0.0, 10000.0
p = [5.0, 0.1, 3.0, 2.0, 0.4, 0.01]

# Solve the ODE
# NOTE: t_eval resolution can be lowered if memory/time is an issue.
t_eval = np.linspace(t0, tf, 20001)  # 20001 points -> step 0.5
sol = solve_ivp(lambda t, y: parameterized_model(t, y, p),
                (t0, tf), u0, t_eval=t_eval, vectorized=False, rtol=1e-6, atol=1e-9)

if not sol.success:
    raise RuntimeError("ODE solver failed: " + str(sol.message))

t = sol.t
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]

# Create figures similar to your Julia layout
fig, axes = plt.subplots(3, 1, figsize=(5, 9), constrained_layout=False)

# Plot settings common
xlim = (5000, 6500)

axes[0].plot(t, x, color="black")
axes[0].set_xlim(xlim)
axes[0].set_ylim(0, 1)
axes[0].set_xlabel("time")
axes[0].set_ylabel("x")
axes[0].set_title("a", loc="left")
axes[0].grid(False)

axes[1].plot(t, y, color="black")
axes[1].set_xlim(xlim)
axes[1].set_ylim(0, 0.5)
axes[1].set_xlabel("time")
axes[1].set_ylabel("y")
axes[1].set_title("b", loc="left")
axes[1].grid(False)

axes[2].plot(t, z, color="black")
axes[2].set_xlim(xlim)
axes[2].set_ylim(7, 10.5)
axes[2].set_xlabel("time")
axes[2].set_ylabel("z")
axes[2].set_title("c", loc="left")
axes[2].grid(False)

# Aesthetic adjustments similar to your Julia call
for ax in axes:
    ax.tick_params(labelsize=8)
    ax.xaxis.label.set_size(10)
    ax.yaxis.label.set_size(10)

plt.subplots_adjust(hspace=0.25, left=0.12, right=0.95, top=0.96, bottom=0.06)

# Ensure output directory exists then save
out_dir = os.path.join("..", "article", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig2.pdf")
fig.savefig(out_path, bbox_inches="tight")
print(f"Saved figure to: {out_path}")
