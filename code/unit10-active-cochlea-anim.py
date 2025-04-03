import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

N = 1000 
L = 35e-3
H = 1e-3
Hb = 7e-6
rho = 1000

dx = L/N
x = np.arange(0,L,dx)


k1 = 2.2e9*np.exp(-300*x) # kg/m^2/s^2
m1 = 3e-2 # kg/m^2
c1 = 60 + 6700*np.exp(-150*x) # kg/m^2/s
k2 = 1.4e7*np.exp(-330*x) # kg/m^2/s^2
c2 = 44.0*np.exp(-165*x) # kg/m^2/s
m2 = 5e-3 # kg/m^2
k3 = 2.0e7*np.exp(-300*x) # kg/m^2/s^2
c3 = 8*np.exp(-60*x) # kg/m^2/s
k4 = 1.15e9*np.exp(-300*x) # kg/m^2/s^2
c4 = 4400.0*np.exp(-150*x) # kg/m^2/s

def solve_active_cochlea(f,  gamma):

    w = 2*np.pi*f
    Z1 = 1j*w*m1 + c1 + 1/1j/w*k1
    Z2 = 1j*w*m2 + c2 + 1/1j/w*k2
    Z3 = c3 + 1/1j/w*k3
    Z4 = c4 + 1/1j/w*k4

    Z = Z1 + Z2*(Z3 - gamma*Z4)/(Z2 + Z3)
    Y = 1/Z

    ldx2 = 2*rho*1j*w*Y/H*dx**2

    A  = np.zeros((N,N),dtype=np.complex128)
    A[0,0] = -2 - ldx2[0]
    A[0,1] = 2
    for nn in range(1,N-1):
        A[nn,nn-1] = 1
        A[nn,nn] = -2 - ldx2[nn]
        A[nn,nn+1] = 1
    A[-1,-1] = 1
    A[-1,-1] = -2 - ldx2[-1]

    us = 1
    b = np.zeros(N, dtype=np.complex128)
    b[0] = -4*1j*w*rho*us*dx

    p = np.linalg.solve(A,b)
    v = Y*p

    return v, p

f = 1000
gamma = 1
t_max = 10e-3
v, p = solve_active_cochlea(f, gamma)

# グラフの初期設定
fig, ax = plt.subplots()
ln, = ax.plot([], [], 'tab:blue')

# グラフの表示範囲を設定
ax.set_xlim(0, 35)

vmax = np.max(np.abs(v))
ax.set_ylim(-vmax*1.1, vmax*1.1)
ax.grid()

# 初期化関数
def init():
    ln.set_data(x*1e3, np.zeros(N))
    return ln,

# フレーム更新関数
def update(frame):

    t = frame
    y = v * np.exp(2j*np.pi*f*t)
    ln.set_data(x*1e3, y.real)
    return ln,

# アニメーション作成
ani = FuncAnimation(fig, update, frames=np.linspace(0, t_max, 128),
                    init_func=init, interval=50)

plt.show()