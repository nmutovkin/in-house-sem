"""Independent reference solutions (semi-analytic), used to verify the
simulator and to compare synthetic data with theory."""

from .borehole import borehole_axis_pressure, borehole_dipole_pressure, borehole_mode_slowness, layer_matrix_size

__all__ = ["borehole_axis_pressure", "borehole_dipole_pressure", "borehole_mode_slowness", "layer_matrix_size"]
