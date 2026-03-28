"""
Unified figure style for the solar forecasting paper.
Import this module before plotting to get consistent colors, fonts, and layout.

Usage:
    from figure_style import *
    apply_style()
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

# ── Province palette ──
PROVINCE_COLORS = {
    'caceres': '#C0392B',   # warm red
    'cadiz':   '#2471A3',   # steel blue
    'zaragoza':'#27AE60',   # green
}
PROVINCE_DISPLAY = {
    'caceres': 'Cáceres',
    'cadiz':   'Cádiz',
    'zaragoza':'Zaragoza',
}

# ── Model palette ──
MODEL_COLORS = {
    'TFT Standard':     '#5B9BD5',  # soft blue
    'TFT NWP':          '#1F4E79',  # dark navy
    'TFT Perfect':      '#70AD47',  # muted green
    'XGBoost Standard': '#ED7D31',  # orange
    'XGBoost NWP':      '#C00000',  # dark red
    'Stacking':         '#7030A0',  # purple
}

# Short aliases for tight legends
MODEL_SHORT = {
    'TFT Standard':     'TFT Std.',
    'TFT NWP':          'TFT NWP',
    'TFT Perfect':      'TFT Perfect',
    'XGBoost Standard': 'XGB Std.',
    'XGBoost NWP':      'XGB NWP',
    'Stacking':         'Stacking',
}

# ── Ordered lists for consistent plotting order ──
REGION_ORDER = ['caceres', 'cadiz', 'zaragoza']
MODEL_ORDER = [
    'TFT Standard', 'TFT NWP', 'TFT Perfect',
    'XGBoost Standard', 'XGBoost NWP', 'Stacking',
]

# ── Heatmap colormap ──
HEATMAP_CMAP = 'YlOrRd'

# ── Figure output settings ──
FIG_DPI = 300
FIG_FORMAT = ['png', 'pdf']


def apply_style():
    """Apply global matplotlib style for paper figures."""
    plt.rcParams.update({
        # Font
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.titleweight': 'bold',
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,

        # Layout
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,

        # Lines
        'lines.linewidth': 1.5,

        # Figure
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.dpi': FIG_DPI,
    })


def save_fig(fig, name, folder='figures'):
    """Save figure as PNG and PDF."""
    import os
    os.makedirs(folder, exist_ok=True)
    for fmt in FIG_FORMAT:
        path = os.path.join(folder, f'{name}.{fmt}')
        fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
    print(f'Saved: {folder}/{name}.png + .pdf')
