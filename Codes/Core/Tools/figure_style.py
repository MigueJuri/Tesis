import matplotlib.pyplot as plt
from cycler import cycler

def set_figure_style():
    """
    Configures global matplotlib settings optimized for a thesis presentation.
    Adapted for the 'Corporate Blue' theme.
    """
    # Define Corporate Blue theme colors
    text_color = '#1C3B5E'      # Navy Blue for text, ticks, and axes spines
    grid_color = '#F4F7FA'   # Light steel blue for subtle grid lines
    
    # Primary, Secondary, Tertiary blues, and a lighter highlight for plotting data
    theme_colors = ['#0056B3', '#4A90E2', '#00B4D8', '#1C3B5E', '#8ECAE6']

    plt.rcParams.update({
        # Figure properties
        'figure.figsize': (10, 5.625),  # 16:9 aspect ratio fits Google Slides perfectly
        'figure.dpi': 300,              # 300 DPI is ideal for screen presentations
        'figure.facecolor': 'none',     # Transparent background lets the slide's #F4F7FA show
        'axes.facecolor': 'none',       
        'savefig.transparent': True,    
        
        # Font properties (Switched to sans-serif for screen readability)
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'text.color': text_color,
        'axes.labelcolor': text_color,
        'xtick.color': text_color,
        'ytick.color': text_color,
        
        # Increased font sizes for presentation visibility
        'font.size': 14,                
        'axes.titlesize': 18,           
        'axes.labelsize': 16,           
        'xtick.labelsize': 12,          
        'ytick.labelsize': 12,          
        'legend.fontsize': 12,          
        
        # Line, marker, and color cycle properties
        'axes.prop_cycle': cycler(color=theme_colors),
        'lines.linewidth': 2.5,         # Thicker lines for projector visibility
        'lines.markersize': 8,          
        
        # Axes and grid properties
        'axes.linewidth': 1.5,          
        'axes.edgecolor': text_color,   # Binds the axes border to the theme's dark blue
        'axes.grid': True,              
        'grid.color': grid_color,       # Tinted grid instead of default gray/black
        'grid.alpha': 0.6,              
        'grid.linestyle': '--'          
    })