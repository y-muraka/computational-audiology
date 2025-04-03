import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    plt.rcParams['font.size'] = 14  # 基本フォントサイズ
    plt.rcParams['axes.titlesize'] = 16  # タイトルのフォントサイズ
    plt.rcParams['axes.labelsize'] = 15  # 軸ラベルのフォントサイズ
    plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りラベルのフォントサイズ
    plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りラベルのフォントサイズ
    plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ
    plt.rcParams['figure.titlesize'] = 18  # 図全体のタイトルのフォントサイズ

    # First plot: Effect of different gamma values at fixed frequency
    f = 1000  # Fixed frequency of 1000 Hz
    plt.figure(figsize=(12, 8))

    # Create 2x1 subplot structure
    for gamma in np.linspace(0,1,5):
        v, p = solve_active_cochlea(f, gamma)


        plt.subplot(211)  # First subplot for amplitude
        plt.plot(x*1e3, 20*np.log10(np.abs(v)), label=f'$\gamma$ = {gamma}')

        plt.subplot(212)
        plt.plot(x*1e3, np.unwrap(np.angle(v))/2/np.pi, label=f'$\gamma$ = {gamma}', lw = 2)

    plt.subplot(211)
    plt.xlim([0, 35])
    plt.ylim([-50, 150])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Basilar Membrane Response at {f} Hz with Different $\gamma$')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(212)
    plt.xlim([0, 35])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Phase [cycle]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('cochlea_gamma_comparison.pdf')
    plt.show()

    # Second plot: Effect of different frequencies at fixed Q
    plt.figure(figsize=(12, 8))

    # Create 2x1 subplot structure
    frequencies = [250, 1000, 4000]
    for gamma in [0,1]:
        for f in frequencies:
            v, p = solve_active_cochlea(f, gamma)
            
            if gamma == 0:
                c = 'tab:orange'
            else:
                c = 'tab:blue'

            plt.subplot(211)  # First subplot for amplitude
            plt.plot(x*1e3, 20*np.log10(np.abs(v)), label=f'{f} Hz', lw=2, c=c)

            plt.subplot(212)
            plt.plot(x*1e3, np.unwrap(np.angle(v))/2/np.pi, label=f'{f} Hz', c=c)

    plt.subplot(211)
    plt.xlim([0, 35])
    plt.ylim([-50, 150])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Basilar Membrane Response at with Different Frequencies')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(212)
    plt.xlim([0, 35])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Phase [cycle]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('cochlea_frequency_comparison.pdf')
    plt.show()
