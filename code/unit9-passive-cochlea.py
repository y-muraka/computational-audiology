import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    plt.rcParams['font.size'] = 14  # 基本フォントサイズ
    plt.rcParams['axes.titlesize'] = 16  # タイトルのフォントサイズ
    plt.rcParams['axes.labelsize'] = 15  # 軸ラベルのフォントサイズ
    plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りラベルのフォントサイズ
    plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りラベルのフォントサイズ
    plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ
    plt.rcParams['figure.titlesize'] = 18  # 図全体のタイトルのフォントサイズ

    # First plot: Effect of different Q values at fixed frequency
    f = 1000  # Fixed frequency of 1000 Hz
    plt.figure(figsize=(12, 8))

    # Create 2x1 subplot structure
    for Qn in [3, 30]:
        v, p = solve_passive_cochlea(f, Qn)


        plt.subplot(211)  # First subplot for amplitude
        plt.plot(x*1e3, 20*np.log10(np.abs(v)), label=f'Q = {Qn}')

        plt.subplot(212)
        plt.plot(x*1e3, np.unwrap(np.angle(v))/2/np.pi, label=f'Q = {Qn}', lw = 2)

    plt.subplot(211)
    plt.xlim([0, 35])
    plt.ylim([-50, 50])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Basilar Membrane Response at {f} Hz with Different Q Values')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(212)
    plt.xlim([0, 35])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Phase [cycle]')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.savefig('cochlea_Q_comparison.pdf')
    plt.show()

    # Second plot: Effect of different frequencies at fixed Q
    Qn = 3  # Fixed Q value of 3
    plt.figure(figsize=(12, 8))

    # Create 2x1 subplot structure
    frequencies = [250, 1000, 4000]
    for f in frequencies:
        v, p = solve_passive_cochlea(f, Qn)
        
        plt.subplot(211)  # First subplot for amplitude
        plt.plot(x*1e3, 20*np.log10(np.abs(v)), label=f'{f} Hz', lw=2)

        plt.subplot(212)
        plt.plot(x*1e3, np.unwrap(np.angle(v))/2/np.pi, label=f'{f} Hz')

    plt.subplot(211)
    plt.xlim([0, 35])
    plt.ylim([-50, 50])
    plt.xlabel('Cochlear location $x$ [mm]')
    plt.ylabel('Amplitude [dB]')
    plt.title(f'Basilar Membrane Response at Q = {Qn} with Different Frequencies')
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
