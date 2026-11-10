"""Response calibration, separated from GSAS profile shape and acquisition."""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.optimize import nnls
from .physics import Cylinder, bin_widths, cylinder_attenuation, hb2a_cross_section, positive


def verify_calibration_asset(path, base):
    base=Path(base).resolve();path=Path(path).resolve()
    if not path.is_relative_to(base):raise ValueError('Calibration asset is outside the packaged data')
    provenance=json.loads((base.parent/'source-provenance.json').read_text())
    name='data/'+str(path.relative_to(base))
    rows=[r for r in provenance['sources'] if r['destination']==name]
    if len(rows)!=1 or hashlib.sha256(path.read_bytes()).hexdigest()!=rows[0].get('packaged_sha256',rows[0]['source_sha256']):
        raise ValueError('The historical calibration asset is missing or changed: '+name)


def cross_sections(metadata, wavelength, table):
    """Per reference cell, with consistent isotope amplitudes and disorder."""
    from .isotopes import bound_cross_sections, isotope_key, resonant
    wavelength = np.asarray(wavelength, float)
    if np.any(wavelength <= 0) or not np.isfinite(wavelength).all():
        raise ValueError('Invalid wavelength')
    absorption, incoherent, scattering = (np.zeros_like(wavelength) for _ in range(3))
    resonance_elements = []
    for element, number in metadata['composition'].items():
        isotope = metadata.get('isotope_selection', {}).get(element, 'Nat. Abund.')
        bl = metadata.get('neutron_table', {}).get(element, [None, {}])[1]
        if 'SL' in bl:
            local_absorption, local_incoherent, local_scattering = bound_cross_sections(
                element, isotope, bl, wavelength, table)
            absorption += number*local_absorption
            incoherent += number*local_incoherent
            scattering += number*local_scattering
        else:
            if not isinstance(isotope, str):
                raise ValueError('Isotope mixtures require explicit neutron component records')
            xs = table['elements'][isotope_key(element, isotope)]
            absorption += number*xs['absorption_2200_b']*wavelength/table['reference_wavelength_a']
            incoherent += number*xs['incoherent_b']
            scattering += number*xs['scattering_b']
        if resonant(bl):
            resonance_elements.append(element)
    return absorption, incoherent, scattering, resonance_elements


def fit_reference_response(x, observed, sigma, calculated, exposure, *, mask=None, degree=3):
    """Fit scale + nonnegative Bernstein background to a measured standard.

    Returns a reduced-intensity scale and effective noise conversion. The latter
    is NOT a raw-count calibration if reduction covariance/reference variance is
    unknown. Callers must preserve that qualification in their calibration file.
    """
    from scipy.special import comb
    x, y, error, model = map(lambda v:np.asarray(v, float), (x, observed, sigma, calculated))
    if not (x.shape == y.shape == error.shape == model.shape) or x.ndim != 1:
        raise ValueError('Reference arrays must have matching vector shapes')
    exposure = positive(exposure, 'reference exposure')
    good = np.isfinite(x+y+error+model) & (error > 0) & (y > 0) & (model >= 0)
    if mask is not None:
        good &= np.asarray(mask, bool)
    if np.count_nonzero(good) < degree+5 or np.max(model[good]) <= 0:
        raise ValueError('Insufficient reference information')
    t = (x-x.min())/(x.max()-x.min())
    poly = np.array([comb(degree,k)*t**k*(1-t)**(degree-k) for k in range(degree+1)]).T
    model_max = model[good].max()
    design = np.column_stack((model/model_max, poly))
    coeff, _ = nnls(design[good]/error[good,None], y[good]/error[good])
    prediction = design@coeff
    return dict(scale=float(coeff[0]/model_max), background=poly@coeff[1:], prediction=prediction,
                effective_response=np.divide(y, error**2*exposure, out=np.zeros_like(y), where=good),
                chi2_per_dof=float(np.sum(((y[good]-prediction[good])/error[good])**2)/(good.sum()-len(coeff))),
                good=good, status='effective_reduced_variance_only_not_raw_count_calibration')


def powgen_attenuation(config, metadata, arrays, sample, table):
    """Approximate focused-bin attenuation over allowed real angles/wavelengths.

    A focused 90-degree TOF coordinate is NOT an incident wavelength. Integrate
    a declared uniform-solid-angle/flat-band kernel instead. A detector-weighted
    response kernel is needed for a validated strongly absorbing sample.
    """
    geometry= Cylinder(**sample)
    q=geometry.quantities(metadata['cell_mass_da'],metadata['cell_volume_a3'])
    ds=np.geomspace(arrays['d'].min(), arrays['d'].max(), 160)
    lo,hi=config['wavelength_band_a']
    result=[]
    for d in ds:
        lower=max(8., float(np.rad2deg(2*np.arcsin(min(1.,lo/(2*d))))))
        upper=min(170., float(np.rad2deg(2*np.arcsin(min(1.,hi/(2*d))))))
        if upper<=lower or lo>2*d:
            result.append(0.)
            continue
        a=np.linspace(lower,upper,41)
        wave=2*d*np.sin(np.deg2rad(a/2))
        abs_xs,inc,scatt,resonances=cross_sections(metadata,wave,table)
        mu=q['cells_per_cm3']*(abs_xs+scatt)*1e-24
        weights=np.sin(np.deg2rad(a))
        transmission=cylinder_attenuation(geometry,a,mu,quadrature=12)
        result.append(float(np.average(transmission,weights=weights)))
    return np.interp(arrays['d'],ds,result),q


def powgen_response(config, metadata, arrays, sample, settings, base):
    path=Path(base)/settings['calibration']
    verify_calibration_asset(path,base)
    calibration=json.loads(path.read_text())
    if calibration['source_bank']!=config['source_bank']:
        raise ValueError('Response belongs to a different wavelength configuration')
    if calibration['kind']!='PNT' or calibration['guide']!='HighRes' or calibration['frequency_hz']!=60:
        raise ValueError('Incompatible POWGEN response configuration')
    if sample.get('wall_thickness_cm',0):
        raise ValueError('POWGEN can-wall corrections require a measured response kernel; do not silently ignore them')
    table_path=Path(base)/'calibration/cross-sections.json';verify_calibration_asset(table_path,base)
    table=json.loads(table_path.read_text())
    transmission,q=powgen_attenuation(config,metadata,arrays,sample,table)
    array_path=path.parent/calibration['arrays']
    verify_calibration_asset(array_path,base)
    data=np.load(array_path)
    # A bin outside the reference range cannot acquire a borrowed count rate.
    d=arrays['d'];low,high=data['d'][0],data['d'][-1]
    valid=(d>=low)&(d<=high)&(transmission>0)
    if valid.mean()<.95:
        raise ValueError('More than 5% of requested d range is outside calibrated coverage')
    # End-bin clipping is explicitly recorded; no scientific use beyond coverage.
    if not valid.all():
        raise ValueError(f'Trim d range to the measured response [{low:.8g}, {high:.8g}] A and nonzero geometry coverage')
    response=np.interp(d,data['d'],data['effective_response_per_uah'])
    # Effective count response scales with bin width: rebinned density retains
    # its mean and gains counts. The calibration array contains native widths.
    response*=bin_widths(arrays['x'])/np.interp(d,data['d'],data['bin_width_us'])
    response*=positive(settings.get('neutron_yield_ratio',1.),'neutron yield ratio')
    n=q['unit_cells']*q['vertical_illumination_fraction']
    scale=calibration['gsas_scale']/calibration['illuminated_unit_cells']*n
    standard_t=np.interp(d,data['d'],data['standard_transmission'])
    absorption_ratio=transmission/standard_t
    nuclear=arrays['nuclear']*scale*absorption_ratio
    magnetic=arrays['magnetic']*scale*absorption_ratio
    if 'angular_nuclear' in arrays:
        angles=arrays['kernel_angles_deg']
        lo,hi=config['wavelength_band_a']
        # Integrate the overlap of each angular quadrature cell with the band.
        # This prevents artificial holes/jumps from a binary node-in-band mask.
        edges=np.r_[8.,(angles[1:]+angles[:-1])/2,170.]
        low_angle=np.maximum(8.,np.rad2deg(2*np.arcsin(np.minimum(1.,lo/(2*d)))))
        high_angle=np.minimum(170.,np.rad2deg(2*np.arcsin(np.minimum(1.,hi/(2*d)))))
        left=np.maximum(edges[:-1,None],low_angle)
        right=np.minimum(edges[1:,None],high_angle)
        weights=np.where(right>left,np.cos(np.deg2rad(left))-np.cos(np.deg2rad(right)),0.)
        if np.any(weights.sum(axis=0)<=0):
            raise ValueError('Angular kernel leaves uncovered bins; refine the kernel')
        weights/=weights.sum(axis=0)
        attenuation=[]
        geometry=Cylinder(**sample)
        ds=np.geomspace(d.min(),d.max(),220)
        for angle in angles:
            wavelengths=2*ds*np.sin(np.deg2rad(angle/2))
            abs_xs,_,scatt,_=cross_sections(metadata,wavelengths,table)
            mu=q['cells_per_cm3']*(abs_xs+scatt)*1e-24
            t=cylinder_attenuation(geometry,np.full_like(ds,angle),mu,quadrature=16)
            attenuation.append(np.interp(d,ds,t))
        attenuation=np.asarray(attenuation)
        transmission=np.sum(weights*attenuation,axis=0)
        nuclear=np.sum(arrays['angular_nuclear']*weights*attenuation,axis=0)*scale/standard_t
        magnetic=arrays['magnetic']*transmission*scale/standard_t
    # This background is an empirical standard/environment shape, scaled only
    # by an explicit user factor. It is not a prediction for an arbitrary can.
    background_y=np.interp(d,data['d'],data['background_y'])*float(settings.get('background_multiplier',1.))
    if np.any(background_y<0):raise ValueError('Negative background multiplier')
    status='historical_standard_transfer_effective_noise'
    summary=dict(model='POWGEN standard-referenced reduced response',exposure_unit='microampere_hour',
        calibration_status=status,source_calibration=str(path),source_run=calibration['run'],
        calibration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        calibration_arrays_sha256=hashlib.sha256(array_path.read_bytes()).hexdigest(),
        reference_proton_energy_gev=1.0,
        neutron_yield_ratio=float(settings.get('neutron_yield_ratio',1.)),
        reference_profile_sha256=calibration['profile_sha256'],target_profile_sha256=metadata['profile_sha256'],
        y_unit='reference-reduction intensity density per microsecond',geometry=q,
        gsas_scale=scale,reference_fit_chi2_per_dof=calibration['fit_chi2_per_dof'],
        bound_scattering_model='coherent neutron amplitude plus thermal spin/within-stock incoherence and isotope disorder/1',
        assumptions=['Standard intensity scales with illuminated unit-cell count',
          'Bound scattering uses coherent neutron amplitudes plus tabulated thermal spin/within-stock incoherence; resonant spin incoherence is not energy-resolved',
          'Uniform solid-angle, flat-band approximation for focused absorption',
          'Resonant nuclear structure factors recomputed at real angular quadrature nodes where present',
          'Effective independent count statistics inferred from reduced Y/E^2, not raw-event counts'],
        warnings=['2022B response transferred to requested cycle; current flux/spectrum not validated',
          'Isotope substitution holds the supplied structure/ADPs fixed; isotope effects on dynamics or structure are not predicted',
          'Reference variance may contain vanadium or focusing covariance; 1/exposure extrapolation is approximate',
          'Strong absorption and resonant nuclei require a measured wavelength/angle response kernel',
          'Background is transferred from Si/PAC06; sample incoherent and environment scattering are not separately calibrated'])
    return dict(response=response,rate=response*(nuclear+magnetic),background_rate=response*background_y,
                nuclear=nuclear,magnetic=magnetic,attenuation=transmission),summary
