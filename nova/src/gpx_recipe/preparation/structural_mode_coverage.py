"""Add named native-mode rays without replacing the mixed-mode portfolio."""
from dataclasses import dataclass

import numpy as np

from .structural_mode_directions import from_source as mode_directions
from .structural_sampling import TrajectorySamples,_interval,physical_grams

SCHEMA='radar-additive-native-mode-coverage/v1'
TARGETED_SCHEMA='radar-additive-native-mode-coverage/v2'
SCHEMAS=frozenset({SCHEMA,TARGETED_SCHEMA})
CONTEXTS_PER_MODE=3


@dataclass(frozen=True)
class ModeCoverage:
    samples: TrajectorySamples
    base_examples: int
    labels: tuple
    gauge_axes: tuple
    rays: tuple
    rejected_amplitudes: int
    maximum_reconstruction_error: float
    strong_targets: dict | None = None


def supplement(source,base,*,seed,steps=4,contexts=CONTEXTS_PER_MODE,
               displacement_range=(.0005,.04),trajectory_feasible=None,maximum_attempts=1000):
    """Use the original shared parent contexts for additional complete rays.

    Every non-gauge native axis retains both signs and every amplitude step.
    A rejected amplitude is redrawn on that same axis, sign and parent; no
    easier mode or different parent can replace it. A failed scope fails the
    supplement. Fold assignment, geometry labels and diffraction are absent
    here. Gauge axes remain declared controls represented by the shared nulls.
    """
    if (not isinstance(base,TrajectorySamples) or type(seed)is not int or not 0<=seed<2**32
            or type(steps)is not int or not 1<=steps<=32
            or type(contexts)is not int or not 1<=contexts<=32
            or type(maximum_attempts)is not int or not 1<=maximum_attempts<=100000
            or trajectory_feasible is not None and not callable(trajectory_feasible)):
        raise ValueError('A complete base portfolio and bounded named-mode sampling controls are required')
    nm=source.mode_dimensions;dimension=len(source.radii)
    points=np.asarray(base.points);groups=np.asarray(base.trajectory)
    if (points.ndim!=2 or not len(points) or points.shape[1]!=dimension
            or not np.isfinite(points).all() or groups.shape!=(len(points),)
            or groups.dtype.kind not in 'iu' or np.any(groups<0)
            or any(np.shape(v)!=(len(points),) for v in (base.displacement_rms,base.relative_metric_norm))
            or not np.isfinite(base.displacement_rms).all() or not np.isfinite(base.relative_metric_norm).all()
            or np.any(base.displacement_rms<0) or np.any(base.relative_metric_norm<0)):
        raise ValueError('Named coverage must retain a complete finite physical base portfolio')
    parent_count=int(groups.max())+1
    if set(map(int,groups))!=set(range(parent_count)) or contexts>parent_count:
        raise ValueError('Use distinct existing shared parent contexts for each named mode')
    directions=mode_directions(source);gram,_=physical_grams(source)
    low,high=_interval(displacement_range,'named-mode RMS displacement')
    physical_axes=np.flatnonzero(~directions.gauge_only)
    total=len(points)+len(physical_axes)*contexts*2*steps
    if total>32768 or total*dimension*8>64*1024**2:
        raise ValueError('Shard complete named-mode rays within the existing numerical workspace')
    arrays=[points.copy()];identities=[groups.copy()]
    rms=[np.asarray(base.displacement_rms).copy()];cell=[np.asarray(base.relative_metric_norm).copy()]
    rng=np.random.default_rng(np.random.SeedSequence([seed,0x4e415449]))
    fractions=np.geomspace(.1,1.,steps);rays=[];rejected=0;cursor=len(points)
    for index in physical_axes:
        parents=rng.choice(parent_count,size=contexts,replace=False)
        for parent in parents:
            for sign in (-1,1):
                direction=np.r_[sign*directions.physical[index],np.zeros(dimension-nm)]
                for attempt in range(maximum_attempts):
                    amplitude=float(np.exp(rng.uniform(np.log(low),np.log(high))))
                    block=fractions[:,None]*(direction*amplitude)
                    allowed=bool(np.all(abs(block)<=source.radii))
                    if allowed:
                        values=np.asarray(source.chemistry.log_kernel(block))
                        if (values.shape!=(steps,) or values.dtype.kind!='f'
                                or np.isnan(values).any() or np.isposinf(values).any()):
                            raise ValueError('Named-mode chemistry must return finite values or negative infinity')
                        allowed=bool(np.isfinite(values).all())
                    if allowed and trajectory_feasible is not None:
                        values=np.asarray(trajectory_feasible(int(parent),block))
                        if values.shape!=(steps,) or values.dtype!=bool:
                            raise ValueError('Named-mode parent admission needs one boolean per point')
                        allowed=bool(values.all())
                    if allowed:break
                    rejected+=1
                else:
                    raise ValueError(f'Cannot complete native mode {index}, sign {sign}, parent {int(parent)} inside the declared scope')
                arrays.append(block);identities.append(np.full(steps,parent,np.int64))
                rms.append(np.sqrt(np.einsum('ni,ij,nj->n',block[:,:nm],gram,block[:,:nm])))
                cell.append(np.zeros(steps))
                rays.append(dict(native_mode=int(index),parent=int(parent),sign=sign,
                    rows=[cursor,cursor+steps],maximum_amplitude=amplitude,rejected_amplitudes=attempt))
                cursor+=steps
    values=[np.concatenate(items) for items in (arrays,identities,rms,cell)]
    for value in values:value.flags.writeable=False
    # The original rejection counter describes the unchanged mixed directions;
    # named-axis amplitude rejections have their own explicit accounting.
    samples=TrajectorySamples(*values,base.rejected_directions,cell_coverage=base.cell_coverage)
    return ModeCoverage(samples,len(points),directions.labels,
        tuple(map(int,np.flatnonzero(directions.gauge_only))),tuple(rays),rejected,
        directions.maximum_reconstruction_error)
