"""Atomic synthetic reduced-file exports and exact-count sidecars."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from .physics import bin_widths


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(temp, path)


def import_reduced(project, datafile, profile, kind):
    """Load our exports in GSAS while retaining supplied zero/negative-Y errors.

    The stock XYE importer sets Y<=0 and its weight to zero. This is undesirable
    for low counts or a blank-subtracted observation. Restore the numeric columns
    from the actual exported file, not simulator truth or a model prediction.
    """
    hist=project.add_powder_histogram(str(datafile),str(profile),fmthint='xye' if kind=='PNC' else 'GSAS')
    if kind=='PNC':
        data=np.loadtxt(datafile,comments='#')
    else:
        rows=[]
        for line in Path(datafile).read_text().splitlines():
            vals=line.split()
            if len(vals)!=3:continue
            try:rows.append([float(v) for v in vals])
            except ValueError:continue
        edge_data=np.asarray(rows)
        widths=np.diff(edge_data[:,0])
        data=np.column_stack(((edge_data[1:,0]+edge_data[:-1,0])/2,
                              edge_data[:-1,1]/widths,edge_data[:-1,2]/widths))
    if not np.allclose(hist.getdata('x'),data[:,0],rtol=1e-12,atol=1e-8):
        raise ValueError('GSAS import grid mismatch')
    if np.any(data[:,2]<=0) or not np.isfinite(data).all():
        raise ValueError('Invalid reduced observation')
    hist.data['data'][1][1]=data[:,1].copy()
    hist.data['data'][1][2]=1/data[:,2]**2
    return hist


def export_snapshot(directory, index, x, snapshot, config, metadata, accumulator):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f'look-{index:04d}'
    extension = '.xye' if config['kind']=='PNC' else '.gsa'
    path = directory/(name+extension)
    if path.exists() or (directory/(name+'.json')).exists():
        raise FileExistsError(f'Refusing to overwrite a released cumulative snapshot: {path}')
    y, e = snapshot.y, snapshot.sigma
    if not np.isfinite(y+e).all() or np.any(e <= 0):
        raise ValueError('Nonfinite intensity or invalid uncertainty')
    provenance = dict(schema='virtual-powder-snapshot/v1', synthetic=True, configuration=config['id'],
                      look=index, cumulative_exposure=snapshot.exposure, **metadata)
    write_pattern(path, x, y, e, config, provenance)
    sidecar = directory/(name+'.npz')
    np.savez_compressed(sidecar, x=x, counts=snapshot.counts, increment=snapshot.increment,
        y=y, sigma=e, diagonal_variance=snapshot.diagonal_variance,
        covariance_factors=snapshot.covariance_factors, shared_reference_variance=snapshot.reference_variance,
        response=accumulator.response, response_estimate=accumulator.response_estimate,
        blank_counts=np.array([]) if accumulator.blank_counts is None else accumulator.blank_counts,
        reference_counts=np.array([]) if accumulator.reference_counts is None else accumulator.reference_counts)
    provenance.update(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sufficient_statistics=sidecar.name, sufficient_statistics_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        rng=accumulator.state(), monitor_counts=snapshot.monitor_counts,
        error_convention='sqrt(diagonal_variance + sum(covariance_factors**2)); Poisson variance uses N+0.5',
        snapshot_relationship='Contains every earlier count; not an independent measurement',
        reference_relationship='Blank and response reference are shared between looks')
    write_json(directory/(name+'.json'),provenance)
    return provenance


def export_count_look(directory, index, x, snapshot, config, metadata, campaign):
    """Actual integer counts with conditional predictive errors and shared maps."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    name=f'look-{index:04d}'
    if (directory/(name+'.json')).exists() or (directory/(name+'.npz')).exists():
        raise FileExistsError('A cumulative count look already exists')
    path=directory/(name+'.gsa')
    provenance=dict(schema='powgen-current-count-look/v1',synthetic=True,look=index,
        configuration=config['id'],cumulative_exposure=snapshot.exposure,
        exposure_unit='microampere_hour',**metadata)
    write_pattern(path,x,snapshot.y,snapshot.sigma,config,provenance)
    sidecar=directory/(name+'.npz')
    np.savez_compressed(sidecar,x=x,counts=snapshot.counts,increment=snapshot.increment,
        expected_counts=snapshot.expected_counts,
        raw_predictive_variance=snapshot.exposure*campaign.rate+snapshot.exposure**2*campaign.rate_variance,
        y=snapshot.y,sigma=snapshot.sigma,counting_variance=snapshot.counting_variance,
        shared_reference_variance=snapshot.reference_variance)
    provenance.update(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        sufficient_statistics=sidecar.name,sufficient_statistics_sha256=hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        total_expected_counts=float(snapshot.expected_counts.sum()),total_simulated_counts=int(snapshot.counts.sum()),
        covariance_model='../reference-model.npz',rng=campaign.rng.bit_generator.state,
        error_convention=('Conditional predictive standard deviation: exact standard ON/OFF and can moments with first-order V-response ratio covariance; shared model in reference-model.npz.' if hasattr(campaign,'cal') else
            'Conditional predictive standard deviation: count variance plus finite ON/OFF reference variance; shared covariance operators in reference-model.npz.'),
        snapshot_relationship='Every cumulative look contains all earlier counts and shares the same uncertain reference rates.',
        reduction_relationship='Observed reference estimates are fixed; reference-rate uncertainty is propagated through posterior-predictive counts, not redrawn as independent denominators.')
    write_json(directory/(name+'.json'),provenance)
    return provenance


def write_pattern(path, x, y, e, config, provenance):
    """Write XYE/FXYE columns without assuming an underlying count likelihood."""
    path = Path(path)
    x, y, e = map(lambda a: np.asarray(a, float), (x, y, e))
    if x.ndim != 1 or len(x) < 3 or x.shape != y.shape or x.shape != e.shape:
        raise ValueError("Pattern arrays must be matching vectors with at least three bins")
    if not np.isfinite(x+y+e).all() or np.any(e <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError("Invalid pattern grid, intensity or uncertainty")
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as out:
        if config['kind']=='PNC':
            out.write('# SYNTHETIC cumulative reduced neutron powder diffraction\n')
            out.write('# '+json.dumps(provenance, separators=(',',':'))+'\n')
            np.savetxt(out, np.column_stack((x,y,e)), fmt='%.12g')
        else:
            bank = config.get('source_bank',1)
            # GSAS-II reads TOF X as LEFT BIN EDGES and drops the last row.
            # Solve edges so their arithmetic centers are exactly our X values.
            # Include a terminal edge row; it carries no additional observation.
            edges=np.empty(len(x)+1)
            edges[0]=x[0]-(x[1]-x[0])/2
            for i in range(len(x)):
                edges[i+1]=2*x[i]-edges[i]
            width=np.diff(edges)
            if np.any(width<=0):
                raise ValueError('TOF grid does not admit increasing bin edges')
            out.write('SYNTHETIC POWGEN cumulative reduced diffraction\n')
            out.write('# '+json.dumps(provenance, separators=(',',':'))+'\n')
            out.write('# Instrument: POWGEN\n# with Y multiplied by the bin widths.\n')
            out.write('# Final numeric row is the terminal bin edge, not a measurement.\n')
            out.write(f'BANK {bank} {len(edges)} {len(edges)} SLOG {edges[0]:.12g} {edges[-1]:.12g} {edges[1]/edges[0]-1:.12g} 0 FXYE\n')
            # Keep round-trip binary64 edges: their subtraction defines the
            # recovered density and magnifies decimal rounding at fine bins.
            np.savetxt(out, np.column_stack((edges,np.r_[y*width,0.],np.r_[e*width,1.])), fmt='%.17g')
    os.replace(temp,path)
