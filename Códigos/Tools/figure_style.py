import matplotlib.pyplot as plt

def set_thesis_figure_style():
    """
    Configures global matplotlib settings for thesis figures.
    Includes high DPI, standard academic fonts, and transparent backgrounds.
    """
    plt.rcParams.update({
        # Figure properties (Background transparency applied here)
        'figure.figsize': (12, 6),       # Standard size (width, height) in inches
        'figure.dpi': 700,              # High resolution for printing
        'figure.facecolor': 'none',     # Transparent figure background
        'axes.facecolor': 'none',       # Transparent axes background
        'savefig.transparent': True,    # Ensure saved files retain transparency
        
        # Font properties
        'font.family': 'serif',         # Serif fonts (e.g., Times) are standard for theses
        'font.size': 12,                # Base font size
        'axes.titlesize': 14,           # Title size
        'axes.labelsize': 12,           # Axis label size
        'xtick.labelsize': 10,          # X-axis tick labels
        'ytick.labelsize': 10,          # Y-axis tick labels
        'legend.fontsize': 10,          # Legend font size
        
        # Line and marker properties
        'lines.linewidth': 2.0,         # Thicker lines for visibility
        'lines.markersize': 6,          # Standard marker size
        
        # Axes properties
        'axes.linewidth': 1.2,          # Thicker axes borders
        'axes.grid': True,              # Enable grid for easier data reading
        'grid.alpha': 0.5,              # Grid line transparency
        'grid.linestyle': '--'          # Dashed grid lines
    })
