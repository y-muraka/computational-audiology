import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams['font.size'] = 14  # 基本フォントサイズ
plt.rcParams['axes.titlesize'] = 16  # タイトルのフォントサイズ
plt.rcParams['axes.labelsize'] = 15  # 軸ラベルのフォントサイズ
plt.rcParams['xtick.labelsize'] = 14  # x軸目盛りラベルのフォントサイズ
plt.rcParams['ytick.labelsize'] = 14  # y軸目盛りラベルのフォントサイズ
plt.rcParams['legend.fontsize'] = 14  # 凡例のフォントサイズ
plt.rcParams['figure.titlesize'] = 18  # 図全体のタイトルのフォントサイズ

def euler_method(m, c, k, F0, omega, x0, v0, dt, t_end):
    """
    Numerical solution of a single-degree-of-freedom oscillation system using Euler method
    
    Parameters:
    -----------
    m : float
        Mass
    c : float
        Damping coefficient
    k : float
        Spring constant
    F0 : float
        Force amplitude
    omega : float
        Angular frequency of the force
    x0 : float
        Initial displacement
    v0 : float
        Initial velocity
    dt : float
        Time step
    t_end : float
        End time
        
    Returns:
    --------
    t : ndarray
        Time array
    x : ndarray
        Displacement array
    v : ndarray
        Velocity array
    """
    # Calculate number of time steps
    n_steps = int(t_end / dt) + 1
    
    # Initialize arrays
    t = np.linspace(0, t_end, n_steps)
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    a = np.zeros(n_steps)
    
    # Set initial conditions
    x[0] = x0
    v[0] = v0
    
    # Calculate initial acceleration
    a[0] = (F0 * np.cos(omega * t[0]) - c * v[0] - k * x[0]) / m
    
    # Time integration using Euler method
    for i in range(1, n_steps):
        # Calculate next state from current state
        x[i] = x[i-1] + v[i-1] * dt
        v[i] = v[i-1] + a[i-1] * dt
        
        # Calculate next acceleration
        a[i] = (F0 * np.cos(omega * t[i]) - c * v[i] - k * x[i]) / m
        
    return t, x, v

def analytical_solution(m, c, k, F0, omega, x0, v0, t):
    """
    Analytical solution of a damped single-degree-of-freedom system with sinusoidal force
    
    Parameters:
    -----------
    m : float
        Mass
    c : float
        Damping coefficient
    k : float
        Spring constant
    F0 : float
        Force amplitude
    omega : float
        Angular frequency of the force
    x0 : float
        Initial displacement
    v0 : float
        Initial velocity
    t : ndarray
        Time array
        
    Returns:
    --------
    x : ndarray
        Displacement array
    """
    # System parameters
    omega_n = np.sqrt(k / m)  # Natural angular frequency
    zeta = c / (2 * np.sqrt(m * k))  # Damping ratio
    omega_d = omega_n * np.sqrt(1 - zeta**2)  # Damped angular frequency
    
    # Frequency ratio
    r = omega / omega_n
    
    # Steady-state response amplitude and phase
    X = (F0 / k) / np.sqrt((1 - r**2)**2 + (2 * zeta * r)**2)
    phi = np.arctan2(2 * zeta * r, 1 - r**2)
    
    # Constants for transient response
    A = x0 - X * np.cos(phi)
    B = (v0 + zeta * omega_n * x0 - zeta * omega_n * X * np.cos(phi) + omega * X * np.sin(phi)) / omega_d
    
    # Analytical solution
    transient = np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))
    steady_state = X * np.cos(omega * t - phi)
    
    return transient + steady_state

def main():
    # System parameters
    m = 1.0      # Mass [kg]
    k = 100.0    # Spring constant [N/m]
    
    # Natural angular frequency
    omega_n = np.sqrt(k / m)
    
    # Damping ratios
    zeta = 0.05
    
    # Frequency ratios (resonance and non-resonance)
    r = 0.5
    
    # End time and time step
    t_end = 50.0  # [s]
    dt = 1e-3     # [s]
    
    # Force amplitude
    F0 = 10.0     # [N]
    
    # Initial conditions
    x0 = 0.0      # Initial displacement [m]
    v0 = 0.0      # Initial velocity [m/s]
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(211)
    ax_error = fig.add_subplot(212)

    # Damping coefficient
    c = 2 * zeta * np.sqrt(m * k)
        
    # Angular frequency of the force
    omega = r * omega_n
            
    # Numerical solution using Euler method
    t, x_euler, v_euler = euler_method(m, c, k, F0, omega, x0, v0, dt, t_end)
            
    # Analytical solution
    x_analytical = analytical_solution(m, c, k, F0, omega, x0, v0, t)

   # Calculate error
    error = x_euler - x_analytical
 
   
    # Plot
    ax.plot(t, x_euler, label='Euler Method', c='tab:blue')
    ax.plot(t, x_analytical,  label='Analytical Solution', c='tab:orange')
    ax.set_ylabel('Displacement [m]')
    ax.set_title(f'Damping Ratio $\\zeta$ = {zeta:.2f}, Frequency Ratio $r$ = {r:.1f}')
    ax.legend()
            
    # Error subplot
    ax_error.plot(t, error, c='tab:blue')
    ax_error.set_ylabel('Error [m]')
    ax_error.set_xlabel('Time [s]')
    
    plt.tight_layout()

    plt.savefig('waveform-Euler.pdf')
    plt.show()

    #Error analysis

    dt_values = [1e-3, 1e-4, 1e-5, 1e-6]
    errors = []
    for dt in dt_values:
        # Numerical solution using Euler method
        t, x_euler, v_euler = euler_method(m, c, k, F0, omega, x0, v0, dt, t_end)
                
       # Calculate error
        errors.append(np.abs(x_euler[-1] - x_analytical[-1]))

    plt.figure(figsize=(8, 6))
    plt.loglog(dt_values, errors, 'o-', label='Error', c='tab:blue')
    plt.loglog(dt_values, [dt for dt in dt_values], '--', label='First Order Slope', c='tab:orange')
    plt.xlabel('Time Step dt')
    plt.ylabel('Error')
    plt.title('Error Order Analysis of Euler Method')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('error-Euler.pdf')
    plt.show()


if __name__ == "__main__":
    main()