import numpy as np
import matplotlib.pyplot as plt

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
c4 = 4400.0*np.exp(-150*x) # kg/m^2/s

dt = 2e-6

def get_g(vb, ub, vt, ut):

    gb = (c1+c3)*vb + (k1+k3)*ub - c3*vt - k3*ut
    gt = -c3*vb - k3*ub + (c2+c3)*vt + (k2+k3)*ut

    uc = ub - ut
    vc = vb - vt
    
    gb -= gamma * np.tanh(c4*vc + k4*uc)

    return gb, gt

def solve_time_domain(f):
    Ntime = int(round(f.size))
    T = Ntime * dt

    alpha2 = 4*rho/H/m1

    vb = np.zeros((Ntime,N))
    ub = np.zeros((Ntime,N))
    vt = np.zeros((Ntime,N))
    ut = np.zeros((Ntime,N))

    p = np.zeros((Ntime,N))

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

    iA = np.linalg.inv(A)

    for ii in range(Ntime-1):
     
        # (ii)
        gb, gt = get_g(vb[ii], ub[ii], vt[ii], ut[ii])

        b = -alpha2*gb
        b[0] -= f[ii] * 2/dx
            
        #(iii)
        p[ii] = iA@b

        #(iv)-(v)
        dvb1 = (p[ii]-gb)/m1 
        ub1 = ub[ii] + dt*vb[ii]
        vb1 = vb[ii] + dt*dvb1

        dvt1 = -gt/m2
        ut1 = ut[ii] + dt*vt[ii]
        vt1 = vt[ii] + dt*dvt1    
            
        # 修正子ステップ
        # (ii)
        gb, gt = get_g(vb1, ub1, vt1, ut1) 

        b = -alpha2*gb
        b[0] -= f[ii+1] * 2/dx

        #(iii)
        p1 = iA@b

        #(iv)-(v)
        dvb2 = (p1-gb)/m1
        vb[ii+1] = vb[ii] + dt/2 * (dvb1 + dvb2)
        ub[ii+1] = ub[ii] + dt/2 * (vb1 + vb[ii+1])

        dvt2 = -gt/m2
        vt[ii+1] = vt[ii] + dt/2 * (dvt1 + dvt2)
        ut[ii+1] = ut[ii] + dt/2 * (vt1 + vt[ii+1])

    return vb, ub, p

if __name__ == "__main__":

    g =  0.6
    Td = 50e-3
    tscale = np.arange(0,Td,dt)
    gamma = np.ones(N)*g


    Lps = [0]
    for fp in [250, 1000, 4000]:
        plt.figure()
        sinewave = np.sin(2*np.pi*fp*tscale) * 100

        Twin = 5e-3
        Nwin = int(round(Twin/dt))
        win = np.sin(np.linspace(0,np.pi/2, Nwin))
        sinewave[:Nwin] *= win

        for Lp in Lps:
            print("%dHz %ddB"%(fp, Lp))

            vb, ub, p = solve_time_domain( sinewave ) # Solve

            plt.plot(x*10, 20*np.log10(np.max(np.abs(vb[int(round(vb.shape[0]*9/10)):]), axis=0)*10))
        plt.xlabel('Distance from the stapes [mm]')
        plt.ylabel('BM velocity [dB re 1 mm/s]')
        plt.title('%d Hz'%(fp))
    plt.show()