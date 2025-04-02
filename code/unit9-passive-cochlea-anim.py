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


m = Hb * rho
fn = 165.4*(10**(2.1*((L-x)/0.035)) - 0.88) # Human's Greenwood function 
wn = 2*np.pi*fn
k = m * wn**2

def solve_passive_cochlea(f,  Qn):
    c = wn*m/Qn

    w = 2*np.pi*f
    Y = 1/(1j*w*m + c + k/1j/w)

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
Qn = 3
t_max = 10e-3
v, p = solve_passive_cochlea(f, Qn)

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