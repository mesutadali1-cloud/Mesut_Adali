import os
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.backends.backend_pdf import PdfPages

os.chdir(os.path.dirname(os.path.abspath(__file__)))

m=1000
xi=0.02
T=1
wn=2*np.pi/T
kp=m*wn**2
c=2*xi*m*wn
dy=0.075
fy=kp*dy
k2=0.1*kp

dt = 0.005
ag = np.loadtxt("ground_acc.txt")*9.806
t = np.arange(len(ag)) * dt
p = -m * ag

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
def tangent(param_a):
    if param_a == 0:
        return kp
    return k2

beta = 1/4
gama = 1/2
a1 = m/(beta*dt**2) + gama*c/(beta*dt)
a2 = m/(beta*dt) + (gama/beta-1)*c
a3 = (1/(2*beta )-1)*m + dt*(gama/(2*beta )-1)*c

n = len(p)

x = np.zeros(n)
v = np.zeros(n)
a = np.zeros(n)
fs = np.zeros(n)

state = np.zeros(n, dtype=int)
iters = np.zeros(n, dtype=int)

a[0] = (p[0] - c * v[0] - fs[0]) / m

t0=time.perf_counter()

for i in range(n - 1):
    p_hat= p[i+1]+a1*x[i]+a2*v[i]+a3*a[i]
    x_den = x[i]
    f_den = fs[i]
    st_den = state[i]
    R = p_hat - f_den - a1 * x_den
    R_eski = 0
    j = 0
    while True:
        K = tangent(st_den) + a1
        x_den = x_den + R / K
        df, st_den = bilinear(dy, fy, k2, x[i], fs[i], x_den - x[i], state[i])
        f_den = fs[i]+df                       # adım başındaki kuvvet + df
        R = p_hat-f_den-a1*x_den                             # p_hat - f_den - a1·x_den
        j = j + 1

        if abs(R) < 1e-6:
            break
        if j >= 2 and abs(R) / abs(R_eski) < 0.01:
            break
        if j > 50:
            print("yakinsamadi, adim", i)
            break
        R_eski = R
    x[i + 1] = x_den
    fs[i + 1] = f_den
    state[i + 1] = st_den
    iters[i + 1] = j

    v[i + 1] = gama/(beta*dt)*(x[i+1]-x[i])+(1-gama/beta)*v[i]+dt*(1-gama/(2*beta))*a[i]
    a[i + 1] = (x[i+1]-x[i])/(beta*dt**2)-v[i]/(beta*dt)-(1/(2*beta)-1)*a[i]

sure=time.perf_counter()-t0
print("iteration time:",sure)

akma = state != 0,
akan_adim_sayisi: np.sum(akma)
print("Yielding steps:", np.sum(akma))

pdf = PdfPages("PartA_results.pdf")

plt.figure()
plt.plot(t, x*100)
plt.title("Part A - Displacemnet Time History")
plt.xlabel("Time [s]")
plt.ylabel("Displacement [cm]")
pdf.savefig()

plt.figure()
plt.plot(t, fs)
plt.title("Part A - Spring force Time History")
plt.xlabel("Time [s]")
plt.ylabel("force [N]")
pdf.savefig()

plt.figure()
plt.plot(x*100, fs)
plt.title("Part A - Spring force Hysteresis")
plt.xlabel("Displacement [cm]")
plt.ylabel("force [N]")
pdf.savefig()

plt.figure()
plt.plot(t, a)
plt.title("Part A - Relative acc. Time History")
plt.xlabel("Time [s]")
plt.ylabel("acceleration [m/s2]")
pdf.savefig()

plt.figure()
plt.plot(t, a+ag)
plt.title("Part A - Absolute acceleration Time History")
plt.xlabel("Time [s]")
plt.ylabel("acceleration [m/s2]")
pdf.savefig()

pdf.close()


### PART B

def newmark(m, c, k, p, dt, beta, gamma, x0, v0):
    n = len(p)
    x = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # initial conditions
    x[0] = x0
    v[0] = v0
    a[0] = (p[0] - c * v0 - k * x0) / m

    # effective stiffness and constants
    k_hat = k + gamma / (beta * dt) * c + m / (beta * dt**2)
    A = m / (beta * dt) + gamma / beta * c
    B = m / (2 * beta) + dt * (gamma / (2 * beta) - 1) * c

    for i in range(n - 1):
        dp_hat = (p[i + 1] - p[i]) + A * v[i] + B * a[i]
        dx = dp_hat / k_hat
        dv = gamma / (beta * dt) * dx - gamma / beta * v[i] + dt * (1 - gamma / (2 * beta)) * a[i]
        da = dx / (beta * dt**2) - v[i] / (beta * dt) - a[i] / (2 * beta)

        x[i + 1] = x[i] + dx
        v[i + 1] = v[i] + dv
        a[i + 1] = a[i] + da

    return x, v, a

x_lin, v_lin, a_lin = newmark(m, c, kp, p, dt, beta, gama, 0, 0)
fs_lin = kp * x_lin

print("Max spring force [N]:",np.max(np.abs(fs)), np.max(np.abs(fs_lin)))
print("Max acc. [m/s2]:",np.max(np.abs(a+ag)), np.max(np.abs(ag+a_lin)))
print("Max velocity. [m/s]:",np.max(np.abs(v)), np.max(np.abs(v_lin)))
print("Max displacement [cm]:",np.max(np.abs(x))*100, np.max(np.abs(x_lin))*100)

pdf = PdfPages("PartB_results.pdf")

plt.figure()
plt.plot(t, x * 100, label="Nonlinear")
plt.plot(t, x_lin * 100, "--", label="Linear")
plt.legend()
pdf.savefig()

pdf.close()