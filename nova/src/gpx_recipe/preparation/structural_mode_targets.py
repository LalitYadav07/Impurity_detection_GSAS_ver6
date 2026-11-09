"""Append geometric-strength controls without replacing weak native examples.

The two-percent contact convention describes training coverage, not a
diffraction-detectability threshold. No candidate or unresolvable direction
is removed. A duplicated endpoint remains explicit and is not an independent
training example; the existing cohort deduplicator retains its lineage.
"""
import copy

import numpy as np

from .structural_mode_coverage import ModeCoverage
from .structural_neighbor_change import from_parent
from .structural_sampling import TrajectorySamples,physical_grams

TARGET_FRACTION=.021
MAXIMUM_RMS=.04
STATUSES=frozenset({'retained_strong_endpoint','appended_strong_target',
    'target_not_reached_inside_declared_ray_bound','unresolved_coincident_parent_contacts',
    'unresolved_parent_distance_precision','unresolved_proposed_contact_precision',
    'native_admission_rejected'})


def validate_record(value,rays):
    """Check the closed coverage inventory, never confer native qualification."""
    if (not isinstance(value,dict) or set(value)!={'target_fraction','rays'}
            or type(value['target_fraction'])not in (int,float) or value['target_fraction']!=TARGET_FRACTION
            or not isinstance(value['rays'],list) or len(value['rays'])!=len(rays)):
        raise ValueError('Strong coverage needs every original ray and its fixed geometric target')
    fields={'ray','status','appended_amplitude','maximum_relative_change','strong'}
    for index,row in enumerate(value['rays']):
        if (not isinstance(row,dict) or set(row)!=fields or type(row['ray'])is not int or row['ray']!=index
                or not isinstance(row['status'],str) or row['status'] not in STATUSES
                or type(row['appended_amplitude'])not in (int,float)
                or not np.isfinite(row['appended_amplitude']) or not 0<row['appended_amplitude']<=MAXIMUM_RMS
                or type(row['maximum_relative_change'])not in (int,float)
                or not np.isfinite(row['maximum_relative_change']) or row['maximum_relative_change']<0
                or row['strong'] is not None and type(row['strong'])is not bool):
            raise ValueError('Invalid complete strong-target ray provenance')
        if row['status'] in {'retained_strong_endpoint','appended_strong_target'}:
            if row['strong'] is not True or row['maximum_relative_change']<=.02:
                raise ValueError('A strong endpoint must retain its measured geometric witness')
        if row['strong'] is False and row['maximum_relative_change']>.02:
            raise ValueError('A measured above-threshold change cannot be labelled weak')


def append(source,coverage,context):
    """Add exactly one endpoint per signed axis/context in the original box.

    Missing geometric targets or chemistry rejections repeat the original
    endpoint and retain an explicit reason. Every original amplitude remains
    bit-identical. Technical geometry errors fail the block instead of being
    interpreted as weak or physically forbidden examples.
    """
    if not isinstance(coverage,ModeCoverage) or coverage.strong_targets is not None or context.source is not source:
        raise ValueError('Strong targets require the original coverage and its matching parent conditioner')
    nm=source.mode_dimensions;geometry=source.fast.empirical_constraint.geometry
    points=coverage.samples.points;total=len(points)+len(coverage.rays)
    if total>32768 or total*len(source.radii)*8>64*1024**2:
        raise ValueError('Shard complete strong mode rays within the existing numerical workspace')
    def expand(point):
        theta=source.lift@point
        return geometry.geometry(theta[:nm],theta[nm:])
    xyz,metric=expand(np.zeros(len(source.radii)))
    graph=from_parent(xyz,metric,geometry.occupancy);gram,_=physical_grams(source)
    parents={};arrays=[points[:coverage.base_examples].copy()]
    identities=[coverage.samples.trajectory[:coverage.base_examples].copy()]
    rms=[coverage.samples.displacement_rms[:coverage.base_examples].copy()]
    cell=[coverage.samples.relative_metric_norm[:coverage.base_examples].copy()]
    rays=[];reports=[];cursor=coverage.base_examples
    for index,ray in enumerate(coverage.rays):
        low,high=ray['rows'];parent=ray['parent'];amplitude=ray['maximum_amplitude']
        block=points[low:high];direction=block[-1]/amplitude
        if np.any(direction[nm:]!=0.):raise ValueError('Geometric contact targeting needs a fixed-cell displacement ray')
        if parent not in parents:parents[parent]=expand(context.controls[parent])
        ref_xyz,ref_metric=parents[parent]
        def report(a):
            changed,g=expand(context.point(parent,direction*a))
            return graph.compare(changed,g,reference_coordinates=ref_xyz,reference_metric=ref_metric)
        chosen=amplitude;last=report(chosen)
        if last['strong_neighbor_distance_change'] is True and last['maximum_relative_change']>=TARGET_FRACTION:
            status='retained_strong_endpoint'
        else:
            center=context.controls[parent];limits=np.full(len(direction),np.inf)
            positive=direction>0;negative=direction<0
            limits[positive]=(source.radii[positive]-center[positive])/direction[positive]
            limits[negative]=(-source.radii[negative]-center[negative])/direction[negative]
            cap=float(min(MAXIMUM_RMS,limits.min()))
            if not np.isfinite(cap) or cap<=0:raise ValueError('A retained physical mode has no positive ray interval')
            motion=geometry.mapping@(source.lift@direction)[:nm]
            proposal=graph.first_strong_amplitude(motion,cap,reference_coordinates=ref_xyz,reference_metric=ref_metric)
            status=proposal['status'];target=proposal['amplitude']
            if target is not None:
                allowed=np.asarray(context.feasible(parent,(direction*target)[None,:]))
                if allowed.shape!=(1,) or allowed.dtype!=bool:
                    raise ValueError('Strong-target admission needs one native physical boolean')
                if allowed[0]:
                    chosen=target;last=report(chosen);status='appended_strong_target'
                    if last['strong_neighbor_distance_change'] is not True:
                        raise ValueError('Conditioning invalidated the proposed strong geometric witness')
                else:status='native_admission_rejected'
        extra=direction*chosen
        # Repeat the saved endpoint itself, not division/multiplication that
        # could change its final bits, whenever no new target was appended.
        if status!='appended_strong_target':extra=block[-1].copy()
        arrays.extend([block.copy(),extra[None,:]])
        identities.append(np.full(len(block)+1,parent,np.int64))
        rms.extend([coverage.samples.displacement_rms[low:high].copy(),
            np.asarray([np.sqrt(max(0.,float(extra[:nm]@gram@extra[:nm])))])])
        cell.extend([coverage.samples.relative_metric_norm[low:high].copy(),np.zeros(1)])
        rays.append(dict(copy.deepcopy(ray),rows=[cursor,cursor+len(block)+1]));cursor+=len(block)+1
        reports.append(dict(ray=index,status=status,appended_amplitude=float(chosen),
            maximum_relative_change=last['maximum_relative_change'],strong=last['strong_neighbor_distance_change']))
    values=[np.concatenate(v) for v in (arrays,identities,rms,cell)]
    assert len(values[0])==total
    for value in values:value.flags.writeable=False
    record=dict(target_fraction=TARGET_FRACTION,rays=reports);validate_record(record,rays)
    return ModeCoverage(TrajectorySamples(*values,coverage.samples.rejected_directions,
        cell_coverage=coverage.samples.cell_coverage),coverage.base_examples,
        coverage.labels,coverage.gauge_axes,tuple(rays),coverage.rejected_amplitudes,
        coverage.maximum_reconstruction_error,record)
