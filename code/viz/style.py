"""Shared visual color code for all README figures.

Palette meaning (reuse everywhere):
  blue   → shared S
  orange → private P
  gray   → concat [P;S]
  green / red diverging → signed gaps / Δ (good / bad around 0)
"""

from __future__ import annotations

COLOR_S = '#2C7BB6'       # shared
COLOR_P = '#D95F0E'       # private
COLOR_PS = '#756BB1'      # concat [P;S]
COLOR_GOOD = '#1A9850'
COLOR_BAD = '#D73027'
COLOR_NEUTRAL = '#F7F7F7'
COLOR_GRID = '#E0E0E0'
COLOR_TEXT = '#222222'

METHOD_MARKERS = {
    'MOCSS': 'o',
    'M1 + b_m': 's',
    'VICReg + b_m': 'D',
    'M2 (AE)': '^',
    'M2 + OMIDIENT P': 'v',
    'OMIDIENT (no ortho)': 'P',
    'OMIDIENT (+ortho)': 'X',
}


def apply_style():
    import matplotlib as mpl

    mpl.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#444444',
        'axes.labelcolor': COLOR_TEXT,
        'axes.titleweight': 'bold',
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.color': COLOR_TEXT,
        'ytick.color': COLOR_TEXT,
        'font.size': 10,
        'legend.fontsize': 9,
        'savefig.dpi': 200,
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
