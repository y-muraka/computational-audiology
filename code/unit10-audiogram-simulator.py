import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from scipy.interpolate import interp1d
from datetime import datetime

plt.rcParams['font.size'] = 14  # 基本フォントサイズ
plt.rcParams['axes.titlesize'] = 16  # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 15  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ
plt.rcParams['figure.titlesize'] = 18  # 図全体のタイトルのフォントサイズ


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

def get_audiogram(fp, gamma):

    v1, _ = solve_active_cochlea(fp,  1)
    v, _ = solve_active_cochlea(fp,  gamma)

    gain = 20*np.log10(np.max(np.abs(v))/np.max(np.abs(v1)))

    return gain

class CochlearApp:
    def __init__(self):
        # Frequency band settings
        self.frequencies = [35/7 * i for i in range(8)]
        self.values = np.ones(len(self.frequencies))
        self.x = np.linspace(0,35,N)
        self.gain = np.ones(self.x.size) * 1
        self.f_cubic = interp1d(self.x, self.gain, kind='cubic')
        
        # Create figure
        self.fig = plt.figure(figsize=(12, 6))
        
        # Create main axes for the frequency response graph
        self.ax_graph = plt.axes([0.1, 0.55, 0.8, 0.4])
        
        # Create sliders
        self.sliders = []
        slider_width = 0.8 / len(self.frequencies)
        
        for i, freq in enumerate(self.frequencies):
            # Calculate slider position
            left = 0.1 + i * slider_width
            width = slider_width * 1
            
            # Create slider
            ax_slider = plt.axes([left, 0.05, width, 0.4])
            slider = Slider(
                ax=ax_slider,
                label='',
                valmin=0,
                valmax=1,
                valinit=1,
                orientation='vertical'
            )
            slider.on_changed(self.update)
            self.sliders.append(slider)
        
        # Create buttons
        # Reset button
        self.reset_button = Button(plt.axes([0.9, 0.05, 0.07, 0.04]), 'Reset')
        self.reset_button.on_clicked(self.reset)
        
        # IO button - positioned above Reset button
        self.io_button = Button(plt.axes([0.9, 0.1, 0.07, 0.04]), 'Calc')
        self.io_button.on_clicked(self.io)

        # Initialize graph
        self.ax_graph.set_title('')
        self.ax_graph.set_xlabel('Cochlear location [mm]')
        self.ax_graph.set_ylabel('Gain factor')
        self.line, = self.ax_graph.plot(self.x, [1]*N, 'k--')
        self.line, = self.ax_graph.plot(self.x, self.gain, c='tab:blue')
        self.ax_graph.grid(True)
        self.ax_graph.set_ylim(-0.1, 1.1)
        
        plt.show()

    def update(self, val):
        # Get all slider values
        self.values = np.array([slider.val for slider in self.sliders])
       
        f_cubic = interp1d(self.frequencies, self.values, kind='cubic') 
        self.gain = f_cubic(self.x)
        # Update graph
        self.line.set_ydata(self.gain)
        
        # Redraw graph
        self.fig.canvas.draw_idle()

    def reset(self, event):
        # Reset all sliders
        for slider in self.sliders:
            slider.reset()
        
        # Reset values to zero
        self.values = np.ones(len(self.frequencies))
        self.gain = np.ones(N) * 1
        # Update graph
        self.line.set_ydata(self.gain)
        
        # Redraw graph
        self.fig.canvas.draw_idle()
        
    def io(self, event):
        try:
            fps = [125, 250, 500, 1000, 2000, 4000, 8000]

            fig = plt.figure(figsize=(10, 8))
            
            ax1 = fig.add_subplot(211)

            ax1.plot(self.x, self.gain, lw = 2)
            
            ax1.set_ylabel('$\gamma$')
            ax1.set_xlabel('Cochlear location [mm]')
            ax1.set_xlim([0, 35])
            ax1.set_ylim([-0.1, 1.1])
            
            ax2 = fig.add_subplot(212)

            gain = []
            for mm in range(len(fps)):
                gain.append(get_audiogram(fps[mm], self.gain))
            ax2.semilogx(fps, gain, lw = 2)
            
            ax2.set_ylabel('[dB]')
            ax2.set_xlabel('Frequency [Hz]')
            ax2.set_xlim([0, 10000])
            ax2.set_ylim([-130, 5])

            ax1.set_title("Simulated audiogram")

            plt.savefig('audiogram.pdf')
            plt.show()
            
            # Reset title after 2 seconds
            self.fig.canvas.draw_idle()

        except Exception as e:
            # Show error message if save fails
            self.ax_graph.set_title(f'Error: {str(e)}', color='red')
            self.fig.canvas.draw_idle()
            plt.pause(2)
            self.ax_graph.set_title('')
            self.fig.canvas.draw_idle()

if __name__ == '__main__':
    app = CochlearApp()