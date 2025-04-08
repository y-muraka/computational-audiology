import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg 


plt.rcParams['font.size'] = 14  # 基本フォントサイズ
plt.rcParams['axes.titlesize'] = 16  # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 15  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ
plt.rcParams['figure.titlesize'] = 18  # 図全体のタイトルのフォントサイズ

N = 400 
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
c4 = 4400.0*np.exp(-150*x)  # kg/m^2/s

dt = 2e-6

alpha2 = 4*rho/H/m1

A = np.zeros((N,N))
A[0,0] = -2 - alpha2*dx**2
A[0,1] = 2
for mm in range(1,N-1):
    A[mm,mm-1] = 1
    A[mm,mm] = -2  - alpha2*dx**2
    A[mm,mm+1] = 1
A[-1,-2] = 1
A[-1,-1] = -2  - alpha2*dx**2
A /= dx**2

lu, piv = linalg.lu_factor(A)

def get_g(vb, ub, vt, ut):
    gb = (c1+c3)*vb + (k1+k3)*ub - c3*vt - k3*ut
    gt = -c3*vb - k3*ub + (c2+c3)*vt + (k2+k3)*ut

    uc = ub - ut
    vc = vb - vt
    
    gb -= gamma * np.tanh(c4*vc + k4*uc)

    return gb, gt

def solve_time_domain(f):
    Ntime = int(round(f.size))

    vb = np.zeros((Ntime,N))
    ub = np.zeros((Ntime,N))
    vt = np.zeros((Ntime,N))
    ut = np.zeros((Ntime,N))

    p = np.zeros((Ntime,N))

    for ii in range(Ntime-1):
        # (ii)
        gb, gt = get_g(vb[ii], ub[ii], vt[ii], ut[ii])

        b = -alpha2*gb
        b[0] -= f[ii] * 2/dx
            
        #(iii) - LU分解を使用して連立方程式を解く
        p[ii] = linalg.lu_solve((lu, piv), b)

        #(iv)-(v)
        dvb1 = (p[ii]-gb)/m1 
        ub1 = ub[ii] + dt*vb[ii]
        vb1 = vb[ii] + dt*dvb1

        dvt1 = -gt/m2
        ut1 = ut[ii] + dt*vt[ii]
        vt1 = vt[ii] + dt*dvt1    
            
        # (ii)
        gb, gt = get_g(vb1, ub1, vt1, ut1) 

        b = -alpha2*gb
        b[0] -= f[ii+1] * 2/dx

        #(iii) - LU分解を使用して連立方程式を解く
        p1 = linalg.lu_solve((lu, piv), b)

        #(iv)-(v)
        dvb2 = (p1-gb)/m1
        dvt2 = -gt/m2

        ub[ii+1] = ub[ii] + dt/2*(vb[ii] + 2*vb1)
        vb[ii+1] = vb[ii] + dt/2*(dvb1 + 2*dvb2) 
        ut[ii+1] = ut[ii] + dt/2*(vt[ii] + vt1)
        vt[ii+1] = vt[ii] + dt/2*(dvt1 + dvt2)


    
    return vb, ub, p

if __name__ == "__main__":

    for fp in [250, 1000, 4000]:
        g = 0.65
        Td = 50e-3
        tscale = np.arange(0,Td,dt/2)
        gamma = np.ones(N)*g

        Lp = range(80,-10,-20)
        for nn in range(len(Lp)):
            A = 10**(Lp[nn]/20) 
            sinewave = A * np.sin(2*np.pi*fp*tscale)

            Twin = 5e-3
            Nwin = int(round(Twin/dt))
            win = np.sin(np.linspace(0,np.pi/2, Nwin))
            sinewave[:Nwin] *= win

            print("%dHz %ddB"%(fp, Lp[nn]))

            vb, ub, p = solve_time_domain(sinewave) # Solve

            vbmax_db =20*np.log10(np.max(np.abs(vb[int(round(vb.shape[0]*9/10)):]), axis=0))
            plt.plot(x*1000, vbmax_db, label=f'{Lp[nn]} dB', lw=2)
        
        if fp == 250 or fp == 1000:
            plt.legend(loc='upper left')
        else:
            plt.legend()

        plt.xlim([0, L*1000])
        plt.ylim([np.max(vbmax_db)-50, np.max(vbmax_db)+80])
        plt.xlabel('Distance from the stapes [mm]')
        plt.ylabel('BM velocity [dB]')
        plt.title(f'{fp} Hz')
        plt.tight_layout()
        plt.savefig(f'bmenv{fp}Hz.pdf')
        plt.clf()
        
        Td = 20e-3
        tscale = np.arange(0,Td,dt)

        g = [0, 0.65]

        Lp = range(0,110,10)

        output = np.zeros((len(g), len(Lp)))
        for mm in range(len(g)):
            gamma = np.ones(N)*g[mm]
            for nn in range(len(Lp)):
                A = 10**(Lp[nn]/20) 
                sinewave = A * np.sin(2*np.pi*fp*tscale)

                Twin = 5e-3
                Nwin = int(round(Twin/dt))
                win = np.sin(np.linspace(0,np.pi/2, Nwin))
                sinewave[:Nwin] *= win

                print("%dHz %ddB"%(fp, Lp[nn]))

                vb, ub, p = solve_time_domain(sinewave) # Solve

                output[mm,nn] =20*np.log10(np.max(np.abs(vb[int(round(vb.shape[0]*9/10)):])))
        
        plt.plot(Lp, output.T-output[1,0], label=[f'$\gamma$={g[0]}',f'$\gamma$={g[1]}'], lw=2)
        plt.plot(Lp, Lp, '--', label='Linear', lw=2)
        plt.xlim([Lp[0], Lp[-1]])
        plt.ylim([Lp[0], Lp[-1]])
        plt.xlabel('Input intensity [dB]')
        plt.ylabel('BM velocity [dB]')

        plt.title(f'{fp} Hz')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'bmio{fp}Hz.pdf')
        plt.clf()

        loudness = np.array(Lp) + output - output[1]
        plt.plot(Lp,loudness.T, label=['Hearing impaired', 'Normal hearing'], lw=2)
        plt.xlim([Lp[0], Lp[-1]])
        plt.ylim([Lp[0], Lp[-1]])
        plt.xlabel('Input intensity [dB]')
        plt.ylabel('Simulated loudness [dB]')
        plt.legend()
        plt.title(f'{fp} Hz')
        plt.tight_layout()
        plt.savefig(f'Loudness{fp}Hz.pdf')
        plt.clf()
