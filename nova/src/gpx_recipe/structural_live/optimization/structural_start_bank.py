"""Multiscale starting points; all final structural coordinates stay free."""
import numpy as np
from scipy.stats import qmc

def multiscale_starts(modes,cells,*,cell_anchor=None,seed=82519):
    if type(modes)is not int or type(cells)is not int or modes<0 or cells<1:raise ValueError('Valid structural dimensions required')
    anchor=np.zeros(cells) if cell_anchor is None else np.asarray(cell_anchor,float)
    if anchor.shape!=(cells,) or not np.isfinite(anchor).all() or np.any(abs(anchor)>1):raise ValueError('Valid bounded cell anchor required')
    points=[np.r_[np.zeros(modes),anchor]]
    if modes:
        directions=qmc.Sobol(modes,scramble=True,seed=seed).random_base2(4)*2-1
        # Small collective displacements remain feasible in larger structures;
        # wider starts explore the original full domain rather than shrinking it.
        for radius in [.04,.12,.35,.8]:
            points.extend(np.r_[radius*row,anchor] for row in directions)
    for row in qmc.Sobol(cells,scramble=True,seed=seed+1).random_base2(3)*.4-.2:
        points.append(np.r_[np.zeros(modes),np.clip(anchor+row,-1,1)])
    return np.asarray(points)


def joint_cell_starts(modes,cells,*,seed=82519):
    """Add wider lattice and simultaneous displacement/lattice starts.

    These cover more of the same declared box; they never expand the domain
    or restrict final refinement to one/two modes.
    """
    base=multiscale_starts(modes,cells,seed=seed);points=list(base)
    directions=qmc.Sobol(cells,scramble=True,seed=seed+2).random_base2(4)*2-1
    for radius in [.04,.12,.35,.8]:points.extend(np.r_[np.zeros(modes),radius*row] for row in directions)
    if modes:
        directions=qmc.Sobol(modes+cells,scramble=True,seed=seed+3).random_base2(4)*2-1
        for radius in [.12,.35,.8]:points.extend(radius*row for row in directions)
    return np.asarray(points)
