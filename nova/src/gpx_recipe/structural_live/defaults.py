"""Shared structural search defaults; no native or UI dependencies.

Cell coordinates are reciprocal-metric changes normalized by the mean parent
diagonal metric. They are not fractional changes of individual cell lengths.
The saved-project audit separately enforces the reviewed length/angle limits.
"""

DEFAULT_CELL_RADIUS = .01
DEFAULT_FIT_SECONDS = 120.
