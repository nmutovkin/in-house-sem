"""Independent reference solutions (semi-analytic), used to verify the
simulator and to compare synthetic data with theory."""

from .borehole import borehole_axis_pressure, layer_matrix_size

__all__ = ["borehole_axis_pressure", "layer_matrix_size"]
