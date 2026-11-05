import numpy as np
import matplotlib.pyplot as plt



#    param_a: 0 = interior, 1 = top border, -1 = bottom border
# k2=ks

def bilinear(dy, fy, k2, x_ini, f_ini, dx, param_a):
    kp = fy / dy
    qy = fy - k2 * dy
    x_new = x_ini + dx
    f_top = qy + k2 * x_new
    f_bot = -qy + k2 * x_new

    if param_a == 0:
        df = kp * dx
        if f_ini + df > f_top:
            return f_top - f_ini, 1
        if f_ini + df < f_bot:
            return f_bot - f_ini, -1
        return df, 0

    if param_a == 1:                         # top
        if dx >= 0:
            return k2 * dx, 1
        df = kp * dx
        if f_ini + df < f_bot:
            return f_bot - f_ini, -1
        return df, 0

    if param_a == -1:                        # bottom
        if dx <= 0:
            return k2 * dx, -1
        df = kp * dx
        if f_ini + df > f_top:
            return f_top - f_ini, 1
        return df, 0

# TEST
dy, fy, k2 = 1.0, 5.0, 0.25
x = np.array([0, 0.5, 0.75, 1.25, 1.5, 1.0, 1.75, 0.75, 0.25, 0.5, 0.0, -0.5,
              -0.75, -1.5, -1.0, -1.75, -2.0, -2.5, -2.25, -1.75, -1.0, -0.25, 0, 1.0])

f = np.zeros(len(x))
param_a = 0

for i in range(1, len(x)):
    df, param_a = bilinear(dy, fy, k2, x[i-1], f[i-1], x[i]-x[i-1], param_a)
    f[i] = f[i-1] + df

dt = 1.0
t = np.arange(len(x)) * dt


plt.figure()
plt.axhline(0, color="black", lw=1)
plt.axvline(0, color="black", lw=1)
plt.plot(x,f)
plt.title("Part 1 - Hysteresis")
plt.xlabel("Displacement")
plt.ylabel("Force")
plt.show()

plt.figure()
plt.plot(t,f)
plt.title("Part 1 - Time-histories of displacement, Force")
plt.xlabel("Time")
plt.ylabel("Force")
plt.show()