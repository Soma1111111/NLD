import os
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# ----- model (same algebra as your Julia parameterized_model!) -----
def f(u, a, b):
    return a * u / (1.0 + b * u)

def parameterized_model(t, u, p):
    x, y, z = u
    a1, a2, b1, b2, d1, d2 = p
    dx = x * (1.0 - x) - f(x, a1, b1) * y
    dy = f(x, a1, b1) * y - f(y, a2, b2) * z - d1 * y
    dz = f(y, a2, b2) * z - d2 * z
    return [dx, dy, dz]

# ----- parameters and timespan -----
t0, tf = 0.0, 10000.0
p = [5.0, 0.1, 3.0, 2.0, 0.4, 0.01]

# We'll evaluate densely so plotting on 0..500 is smooth
t_eval = np.linspace(t0, tf, 20001)  # ~0.5 step

# ----- series 1 (u0 = [0.77, 0.16, 9.9]) -----
u0_serie1 = [0.77, 0.16, 9.9]
sol1 = solve_ivp(lambda t, y: parameterized_model(t, y, p),
                 (t0, tf), u0_serie1, t_eval=t_eval, rtol=1e-9, atol=1e-12)

if not sol1.success:
    raise RuntimeError("ODE solver failed for series 1: " + sol1.message)

# ----- series 2 (x changed by +0.01) -----
u0_serie2 = [0.78, 0.16, 9.9]
sol2 = solve_ivp(lambda t, y: parameterized_model(t, y, p),
                 (t0, tf), u0_serie2, t_eval=t_eval, rtol=1e-9, atol=1e-12)

if not sol2.success:
    raise RuntimeError("ODE solver failed for series 2: " + sol2.message)

# ----- extract x timeseries and plot interval [0, 500] -----
t = sol1.t
x1 = sol1.y[0, :]
x2 = sol2.y[0, :]

# Clip to t <= 500 for plotting (same xlim as your Julia plot)
mask = t <= 500.0
t_plot = t[mask]
x1_plot = x1[mask]
x2_plot = x2[mask]

# ----- plotting -----
fig, ax = plt.subplots(figsize=(8, 3.5))  # wide-ish figure to match a typical time series look

ax.plot(t_plot, x1_plot, color='black', linewidth=2, linestyle='-')                  # solid black
ax.plot(t_plot, x2_plot, color='darkgrey', linewidth=2, linestyle='--')             # dashed dark grey

ax.set_xlim(0, 500)
ax.set_ylim(0, 1)
ax.set_xlabel("time")
ax.set_ylabel("x")
ax.grid(False)
# no legend (legend = false in Julia)
# frame style: Matplotlib default is boxed; that matches framestyle = :box

# aesthetics similar to your Julia guidefontsize/tickfontsize
ax.xaxis.label.set_size(10)
ax.yaxis.label.set_size(10)
ax.tick_params(axis='both', which='major', labelsize=8, direction='out')

# save
out_dir = os.path.join("..", "article", "figures")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "fig3.pdf")
fig.savefig(out_path, bbox_inches='tight')
print("Saved figure to:", out_path)
