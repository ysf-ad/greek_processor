import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from dataclasses import dataclass

@dataclass
class SSVIParams:
    rho: float    # correlation
    eta: float    # ATM vol level
    gamma: float  # overall variance level
    a: float      # left wing
    b: float      # right wing

def ssvi_variance(k: float, params: SSVIParams) -> float:
    """
    Calculate SSVI total variance
    k: log-moneyness
    """
    # ATM total variance
    w = params.eta
    
    # Left and right wing behavior
    left_wing = np.exp(-params.b * k) if k < 0 else 1
    right_wing = np.exp(params.b * k) if k > 0 else 1
    
    # Core SSVI function with asymmetric wings
    v = w * (1 + params.rho * params.gamma * k + 
             np.sqrt((params.gamma * k + params.rho)**2 + (1 - params.rho**2)))
    
    # Apply wing adjustments
    return v * params.a * (left_wing + right_wing) / 2

def plot_ssvi():
    # Create figure and subplot
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.4)  # Make room for sliders
    
    # Initial parameters
    initial_params = SSVIParams(
        rho=-0.7,    # correlation
        eta=0.2,     # ATM vol level
        gamma=1.0,   # variance level
        a=1.0,       # left wing
        b=1.0        # right wing
    )
    
    # Create x points for plotting
    x_points = np.linspace(-0.5, 0.5, 1000)  # log-moneyness range
    
    def update(val=None):
        # Get current parameter values from sliders
        params = SSVIParams(
            rho=rho_slider.val,
            eta=eta_slider.val,
            gamma=gamma_slider.val,
            a=a_slider.val,
            b=b_slider.val
        )
        
        # Calculate variances
        v_points = [ssvi_variance(k, params) for k in x_points]
        iv_points = [np.sqrt(max(0, v))*100 for v in v_points]  # Convert to IV percentage
        
        # Clear and redraw
        ax.clear()
        ax.plot(x_points*100, iv_points, 'b-', linewidth=2)  # Convert to percentage moneyness
        
        # Add vertical line at ATM
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # Configure axes
        ax.set_xlabel('Log-Moneyness (%)')
        ax.set_ylabel('Implied Volatility (%)')
        ax.set_title('SSVI Volatility Surface')
        ax.grid(True, alpha=0.3)
        
        # Set y-axis limits
        ax.set_ylim(0, 100)
        
        # Force redraw
        fig.canvas.draw_idle()
    
    # Create sliders
    slider_color = 'lightgoldenrodyellow'
    
    rho_ax = plt.axes([0.1, 0.30, 0.65, 0.03])
    rho_slider = Slider(
        ax=rho_ax,
        label='ρ (correlation)',
        valmin=-0.99,
        valmax=-0.01,
        valinit=initial_params.rho,
        color=slider_color
    )
    
    eta_ax = plt.axes([0.1, 0.25, 0.65, 0.03])
    eta_slider = Slider(
        ax=eta_ax,
        label='η (ATM level)',
        valmin=0.01,
        valmax=0.5,
        valinit=initial_params.eta,
        color=slider_color
    )
    
    gamma_ax = plt.axes([0.1, 0.20, 0.65, 0.03])
    gamma_slider = Slider(
        ax=gamma_ax,
        label='γ (variance)',
        valmin=0.1,
        valmax=5.0,
        valinit=initial_params.gamma,
        color=slider_color
    )
    
    a_ax = plt.axes([0.1, 0.15, 0.65, 0.03])
    a_slider = Slider(
        ax=a_ax,
        label='a (level)',
        valmin=0.1,
        valmax=3.0,
        valinit=initial_params.a,
        color=slider_color
    )
    
    b_ax = plt.axes([0.1, 0.10, 0.65, 0.03])
    b_slider = Slider(
        ax=b_ax,
        label='b (wing slope)',
        valmin=0.1,
        valmax=5.0,
        valinit=initial_params.b,
        color=slider_color
    )
    
    # Register update function with each slider
    rho_slider.on_changed(update)
    eta_slider.on_changed(update)
    gamma_slider.on_changed(update)
    a_slider.on_changed(update)
    b_slider.on_changed(update)
    
    # Initial plot
    update()
    
    plt.show()

if __name__ == "__main__":
    plot_ssvi() 