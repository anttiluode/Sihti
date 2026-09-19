"""Sihti -- a sieve for images.

Run a purifier (Sigh's Fourier EQ, or the image's own graph), keep everything
that bleeds out between octave depths, and add it all back.  Which purifier you
use decides what "persistence" means: frequency for Sigh's EQ, grouping (parts,
then objects) for the image's own EQ.
"""
from .color import srgb_to_lab
from .graph import GridGraph, grid_graph, self_affinity, lattice_affinity
from .purifiers import FourierEQ, GraphDiffusion, RebuiltSelfEQ, MedianPurifier, SIGH_PRESETS
from .sieve import sift, octave_depths, log_depths, within_fraction, SieveResult
from .modes import spectral_modes, segments, paint, eigengap_k, align, substrate_share
from .pipeline import analyse, resize_to, lattice_modes

__version__ = "0.1.0"
