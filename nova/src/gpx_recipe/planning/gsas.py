"""GSAS-II forward calculation only: no fitting, counting noise or arbitrary scale.

Run with the tested GSAS-II Python and GSAS-II checkout on PYTHONPATH.
"""
from pathlib import Path
import copy
import hashlib
import json
import math
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pnt_d_grid(grid):
    """Bound the requested logarithmic grid before allocating its coordinates."""
    values=[grid['d_min'],grid['d_max'],grid['relative_step']]
    if (any(isinstance(v,(bool,np.bool_)) or not np.isscalar(v) or not math.isfinite(v) for v in values)
            or not 0<values[0]<values[1] or values[2]<=0):
        raise ValueError('Positive increasing d limits and a finite positive relative step required')
    low,high=np.log(values[0]),np.log(values[1])
    points=(high-low)/values[2]
    if not np.isfinite(points) or not 3<=math.ceil(points)<=50000:
        raise ValueError('Requested grid needs between 3 and 50000 points; use a covered bounded step')
    return np.exp(np.arange(low,high,values[2]))


def calculate(config, material, output, base):
    from gpx_recipe.live.validated_backend import VERSION, corrected_backend, active_identity
    requested = material.get('calculation_backend', VERSION)
    if requested not in ('installed', VERSION):
        raise ValueError('Unknown calculation backend')
    if requested == VERSION and active_identity() is None:
        with corrected_backend():
            return calculate(config, material, output, base)
    if requested == VERSION:
        from .isotopes import neutron_backend
        with neutron_backend(material.get('isotopes', {})) as neutron_state:
            return _calculate(config, material, output, base, neutron_state)
    if any(isinstance(v, dict) for v in material.get('isotopes', {}).values()):
        raise ValueError('Isotope abundance mixtures require the corrected planning backend')
    return _calculate(config, material, output, base)


def _calculate(config, material, output, base, neutron_state=None):
    from GSASII import GSASIIscriptable as G
    from GSASII import GSASIIstrMain as S, GSASIIlattice as L, GSASIIElem as E
    from gpx_recipe.live.gpx_index import install_fresh_gpx_index
    from .mixture import additional_phases, mass_fractions, combine_metadata
    install_fresh_gpx_index()
    base, output = Path(base), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    profile = (base/config['profile']).resolve()
    if config.get('profile_sha256') and sha256(profile) != config['profile_sha256']:
        raise ValueError('Instrument profile hash mismatch')
    gpx = G.G2Project(newgpx=str(output/'forward.gpx'))
    grid = config['grid']
    cw = config['kind'] == 'PNC'
    requested_d = None if cw else _pnt_d_grid(grid)
    hist = gpx.add_simulated_powder_histogram('virtual', str(profile),
        grid['min'] if cw else 1., grid['max'] if cw else 100.,
        # The temporary native TOF range is replaced below. Reusing the final
        # fine step here can exceed native limits before the requested bounded
        # coordinates are installed.
        Tstep=grid['step'] if cw else .01, scale=1.,
        **({'ibank': config['profile_bank']} if config.get('profile_bank') else {}))
    inst = hist.InstrumentParameters
    if not str(inst['Type'][0]).startswith(config['kind']):
        raise ValueError('Profile kind does not match registry')
    from .legacy_tof import restore_profile3
    legacy_translation=restore_profile3(profile,int(inst['Bank'][1]),inst) if not cw else None
    resolution_table=None
    if config.get('resolution_table'):
        from .fullprof_lookup import read_lookup
        if cw or material.get('calculation_backend')=='installed':
            raise ValueError('Numerical resolution tables require the corrected TOF planning backend')
        resolution_table=read_lookup(base/config['resolution_table'])
        if resolution_table.source_sha256!=config.get('resolution_table_sha256'):
            raise ValueError('Resolution table hash mismatch')
        if np.any(resolution_table.table[:,5]!=0):
            raise ValueError('Native reflection enumeration currently requires a zero-shift table')
        if resolution_table.two_theta!=inst['2-theta'][1]:
            raise ValueError('Resolution table and crystallographic focused angle differ')
        resolution_table.parameters(requested_d)
        for key,value in zip(('difC','difA','difB','Zero'),resolution_table.coefficients):
            inst[key][0]=inst[key][1]=value
    if not cw:
        d = requested_d
        x = np.asarray(grid.get('tof', L.Dsp2pos(inst, d)), float)
        if x.ndim!=1 or len(x)<3:
            raise ValueError('A one-dimensional TOF grid with at least three bins is required')
        if not np.isfinite(x).all() or np.any(x<=0) or np.any(np.diff(x)<=0):
            raise ValueError('Instrument calibration does not produce a finite positive monotonic TOF axis over this d range')
        if len(x) > 50000:
            raise ValueError('Requested grid too large; use a coarser relative step')
        hist.data['data'][1] = [x, np.ones_like(x), np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)]
        hist.data['Limits'] = [(float(x[0]), float(x[-1])), [float(x[0]), float(x[-1])]]
    x = np.asarray(hist.data['data'][1][0], float)
    if not cw:
        ds = np.asarray(L.Pos2dsp(inst, x))
        sig = inst['sig-0'][1]+inst['sig-1'][1]*ds**2+inst['sig-2'][1]*ds**4+inst.get('sig-q',[0.,0.])[1]*ds
        extra_parameters=hist.data['Instrument Parameters'][1]
        lookup=extra_parameters.get('Pdabc',extra_parameters.get('pdabc'))
        if lookup is not None:
            if not isinstance(lookup,dict) or not {'d','sig'}<=set(lookup):
                raise ValueError('The legacy profile lookup table requires an explicitly supported native interpretation')
            if {'alp','bet','TOF'} & set(lookup):
                raise ValueError('Lookup-table tail/position parameters require an audited native implementation')
            td,ts=np.asarray(lookup['d'],float),np.asarray(lookup['sig'],float)
            if td.ndim!=1 or td.shape!=ts.shape or len(td)<2 or not np.isfinite(td+ts).all() or np.any(np.diff(td)<=0):
                raise ValueError('Invalid profile lookup table')
            if ds.min()<td[0] or ds.max()>td[-1]:
                raise ValueError('Requested d range exceeds the profile lookup table')
            sig+=np.interp(ds,td,ts)
        beta = inst['beta-0'][1]+inst['beta-1'][1]/ds**4+inst.get('beta-q',[0.,0.])[1]/ds**2
        if (not np.isfinite(sig).all() or not np.isfinite(beta).all()
                or np.any(sig<=0) or np.any(beta<=0) or not np.isfinite(inst['alpha'][1]) or inst['alpha'][1]<=0):
            raise ValueError('TOF profile has nonpositive width/tail coefficients in requested range; restrict the calibrated domain')
        if legacy_translation is not None or material.get('calculation_backend')!='installed':
            gamma=inst['Z'][1]+inst['X'][1]*ds+inst['Y'][1]*ds**2
            if not np.isfinite(gamma).all() or np.any(gamma<0):
                raise ValueError('TOF profile has an invalid Lorentzian width in the requested domain')
    hist.data['data'][0]['Dummy'] = False  # GSAS must never invent its own noise
    hist.data['data'][1][1][:] = 1.
    hist.data['Background'][0][1] = False
    hist.data['Background'][0][3:] = [0.] * (len(hist.data['Background'][0])-3)
    hist.data['Background'][1]['nPeaks'] = 0
    hist.data['Background'][1]['peaksList'] = []
    hist.data['Sample Parameters']['Scale'] = [1., False]
    for name in ('Absorption', 'DisplaceX', 'DisplaceY', 'Shift', 'Transparency'):
        if name in hist.data['Sample Parameters']:
            hist.data['Sample Parameters'][name] = [0., False]
    sources = {}
    adp_normalizations = []
    def import_path(path, role):
        if neutron_state is None:
            return path
        from .cif_adps import prepare_cif
        target = output / (role + '-displacements' + path.suffix)
        derived, record = prepare_cif(path, target)
        if record:
            adp_normalizations.append(dict(record, source_role=role))
        return derived
    if material.get('gpx'):
        path = (base/material['gpx']).resolve()
        sources['gpx'] = dict(path=str(path), sha256=sha256(path))
        original = G.G2Project(str(path))
        for old in original.phases():
            if old.data['General']['Type'] not in ('nuclear', 'magnetic'):
                raise ValueError('Only commensurate nuclear/magnetic phases supported')
            ph = gpx.add_phase(phasename=old.name)
            preserved = copy.deepcopy(old.data)
            preserved['Histograms'] = {}
            preserved.pop('pId', None)
            gpx.data['Phases'][ph.name] = preserved
            gpx.link_histogram_phase(hist, ph.name)
    else:
        path = (base/material['cif']).resolve()
        sources['cif'] = dict(path=str(path), sha256=sha256(path))
        nuclear = gpx.add_phase(str(import_path(path, 'nuclear')), phasename='nuclear', histograms=[hist], fmthint='CIF')
        if material.get('mcif'):
            path = (base/material['mcif']).resolve()
            sources['mcif'] = dict(path=str(path), sha256=sha256(path))
            gpx.add_phase(str(import_path(path, 'magnetic')), phasename='magnetic', histograms=[hist], fmthint='CIF', mag=True)
    phases = gpx.phases()
    nuclear_phases = [p for p in phases if p.data['General']['Type'] == 'nuclear']
    if len(nuclear_phases) != 1:
        raise ValueError('Supply exactly one nuclear parent; declare other phases with explicit mass fractions')
    parent = nuclear_phases[0]
    additions = additional_phases(material.get('additional_phases', []))
    fractions = mass_fractions(additions)
    declared_isotopes = material.get('isotopes', {})
    for phase in phases:
        for element,label in phase.data['General'].get('Isotope',{}).items():
            if label.startswith('RADARmix-') and not isinstance(declared_isotopes.get(element),dict):
                raise ValueError('A prepared isotope-mixture GPX requires its explicit atom fractions')
    def configure_isotopes(selected, include_mixtures=True):
        for phase in selected:
            general = phase.data['General']
            for element, selection in declared_isotopes.items():
                if element in general.get('AtomTypes', []):
                    if not include_mixtures and isinstance(selection, dict):
                        native_options=[label for label in general['Isotopes'][element] if not label.startswith('RADARmix-')]
                        general['Isotope'][element]='Nat. Abund.' if 'Nat. Abund.' in native_options else native_options[0]
                        continue
                    label = neutron_state['native_selections'][element] if neutron_state else selection
                    if label not in general['Isotopes'][element]:
                        raise ValueError('Unsupported isotope selection for mixture atom type '+element+': '+str(selection))
                    general['Isotope'][element] = label
            E.SetupGeneral(phase.data, str(output))
    # Symmetry itself does not depend on isotope abundances. Keep this primary
    # model readable by native symmetry tools without the isotope adapter.
    configure_isotopes(phases, include_mixtures=False)
    # Symmetry discovery operates on the primary model, before additional
    # phases and their mass fractions enter the specimen calculation.
    gpx.update_ids()
    gpx.set_Controls('cycles', 0)
    gpx.save(str(output/'reference-parent.gpx'))
    gpx.filename = str(output/'forward.gpx')
    added_native = []
    for index, row in enumerate(additions, 1):
        path = (base/row['cif']).resolve()
        sources['additional_phase_'+str(index)] = dict(path=str(path), sha256=sha256(path))
        phase = gpx.add_phase(str(import_path(path, 'additional-'+str(index))), phasename='additional_'+str(index), histograms=[hist], fmthint='CIF')
        if phase.data['General']['Type'] != 'nuclear':
            raise ValueError('Additional phases must be nuclear CIFs; magnetic scattering belongs to the primary parent')
        added_native.append(phase)
    configure_isotopes(phases + added_native)
    phases = gpx.phases()
    present_types = {element for phase in phases for element in phase.data['General']['AtomTypes']}
    if set(declared_isotopes)-present_types:
        raise ValueError('Isotope selections refer to absent mixture atom types')
    parent_volume = float(parent.data['General']['Cell'][7])
    cell_multiplier=float(material.get('nuclear_cell_multiplier',1.))
    if not np.isfinite(cell_multiplier) or not 1/64<=cell_multiplier<=64:
        raise ValueError('Reference nuclear-cell multiplier is outside the supported range')
    components = []
    nuclear_phases = [parent] + added_native
    for phase in nuclear_phases:
        gen = phase.data['General']
        components.append(dict(cell_mass_da=float(gen['Mass']), cell_volume_a3=float(gen['Cell'][7]),
            composition={str(k):float(v) for k,v in gen['NoAtoms'].items()}, neutron_table=E.GetBLtable(gen),
            isotope_selection={element:copy.deepcopy(declared_isotopes.get(element, choice))
                               for element,choice in gen.get('Isotope', {}).items()},
            isotope_options={element:[label for label in options if not label.startswith('RADARmix-')]
                             for element,options in gen.get('Isotopes',{}).items()}))
    mixture = combine_metadata(components, fractions, float(parent.data['General']['Mass'])/cell_multiplier)
    nuclear_scales = {phase.name:scale for phase,scale in zip(nuclear_phases,mixture['nuclear_phase_scales'])}
    phase_fractions = {phase.name:fraction for phase,fraction in zip(nuclear_phases,fractions)}
    phase_settings = {phase.name:row for phase,row in zip(added_native,additions)}
    phase_metadata = []
    for phase in phases:
        gen = phase.data['General']
        if gen.get('Modulated') or gen.get('Super'):
            raise ValueError('Incommensurate/modulated phases are outside this simulator')
        hap = phase.data['Histograms'][hist.name]
        hap['Scale'] = [nuclear_scales[phase.name] if gen['Type']=='nuclear' else
                        fractions[0]*parent_volume/float(gen['Cell'][7])/cell_multiplier, False]
        settings = dict(material, **phase_settings.get(phase.name, {}))
        # The third entry is the Lorentzian fraction, not another length.
        hap['Size'][1] = [float(settings.get('crystallite_size_um', 1e6))]*2+[1.]
        hap['Mustrain'][1] = [float(settings.get('microstrain', 0.))]*2+[1.]
        hap['HStrain'][0] = [0.]*len(hap['HStrain'][0])
        hap['Pref.Ori.'][1] = float(material.get('march_dollase', 1.))
        hap['Pref.Ori.'][3] = list(material.get('texture_axis', [0, 0, 1]))
        if gen['Type'] == 'magnetic':
            gen['MagDmin'] = float(np.min(L.Pos2dsp(inst, x)))*.9
            # Preserve allowed orientation and site relations, change amplitudes together.
            factor = float(material.get('moment_multiplier', 1.))
            if not np.isfinite(factor) or factor < 0:
                raise ValueError('moment_multiplier must be finite and nonnegative')
            cx = gen['AtomPtrs'][0]
            for atom in phase.data['Atoms']:
                atom[cx+4:cx+7] = (np.asarray(atom[cx+4:cx+7], float)*factor).tolist()
            if material.get('lande_g') is not None:
                gen['Lande g'] = [float(material['lande_g'])]*len(gen['AtomTypes'])
        phase_metadata.append(dict(name=phase.name, type=gen['Type'], cell=[float(v) for v in gen['Cell'][1:]],
            scale=hap['Scale'][0], mass_fraction=phase_fractions.get(phase.name),
            primary_phase_mass_fraction=fractions[0] if gen['Type']=='magnetic' else None,
            role='primary' if phase.name==parent.name else ('additional' if gen['Type']=='nuclear' else 'primary_magnetic'),
            crystallite_size_um=float(settings.get('crystallite_size_um',1e6)),microstrain=float(settings.get('microstrain',0.)),
            spacegroup=str(gen['SGData'].get('MagSpGrp', gen['SGData'].get('SpGrp')))))
    gpx.update_ids()
    gpx.set_Controls('cycles', 0)
    kernel_runs=[]
    def forward_pattern():
        # Keep the explicit installed-backend diagnostic route. The default
        # planning calculation uses the normalized physical convolution.
        if cw or material.get('calculation_backend')=='installed':
            return np.asarray(S.DoNoFit(str(output/'forward.gpx'),hist.name),float)
        from .tof_backend import physical_tof_forward
        transform=None
        if resolution_table is not None:
            from .fullprof_lookup import native_reflection_pattern
            native_inst={k:float(v[1]) for k,v in inst.items() if k in
                ('sig-0','sig-1','sig-2','sig-q','X','Y','Z')}
            transform=lambda points,blocks:native_reflection_pattern(resolution_table,points,blocks,native_inst)
        with physical_tof_forward(transform) as identity:
            result=np.asarray(S.DoNoFit(str(output/'forward.gpx'),hist.name),float)
        if resolution_table is not None:identity['resolution_table']=resolution_table.metadata()
        kernel_runs.append(identity)
        return result
    curves = {}
    for kind in ('nuclear', 'magnetic'):
        if not any(p.data['General']['Type'] == kind and p.data['Histograms'][hist.name]['Scale'][0]>0 for p in phases):
            curves[kind] = np.zeros_like(x)
            continue
        for phase in phases:
            phase.data['Histograms'][hist.name]['Use'] = (phase.data['General']['Type'] == kind and
                phase.data['Histograms'][hist.name]['Scale'][0]>0)
        gpx.save()
        curve = forward_pattern()
        if not np.isfinite(curve).all() or np.min(curve) < -1e-10*max(1., np.max(curve)):
            raise ValueError('GSAS returned invalid forward intensity')
        curves[kind] = np.maximum(0., curve)
    for phase in phases:
        phase.data['Histograms'][hist.name]['Use'] = True
    gpx.save()
    joint = forward_pattern()
    if not np.allclose(joint, curves['nuclear']+curves['magnetic'], rtol=1e-9, atol=1e-10):
        raise ValueError('Nuclear/magnetic additivity check failed')
    gen = parent.data['General']
    neutron_table = mixture['neutron_table']
    extra = {}
    from .isotopes import resonant
    if not cw and any(resonant(neutron_table[element][1]) for element in mixture['composition']):
        # A focused 90-degree coordinate cannot be used as the physical incident
        # wavelength of resonant nuclei. Recalculate nuclear structure factors
        # at representative real detector angles; the response integrates them.
        angles=np.array([8.,9.,10.,12.,14.,16.,18.,20.,24.,28.,32.,38.,45.,52.,60.,70.,80.,90.,100.,110.,120.,130.,140.,150.,160.,170.])
        angular=[]
        nominal=float(inst['2-theta'][1])
        for phase in phases:
            phase.data['Histograms'][hist.name]['Use']=(phase.data['General']['Type']=='nuclear' and
                phase.data['Histograms'][hist.name]['Scale'][0]>0)
        for angle in angles:
            inst['2-theta'][0]=inst['2-theta'][1]=float(angle)
            gpx.save()
            part=forward_pattern()
            if not np.isfinite(part).all() or np.min(part)<-1e-10:
                raise ValueError('Invalid resonant-angle GSAS profile')
            angular.append(np.maximum(0,part)*np.sin(np.deg2rad(nominal/2))/np.sin(np.deg2rad(angle/2)))
        extra=dict(kernel_angles_deg=angles,angular_nuclear=np.asarray(angular))
        inst['2-theta'][0]=inst['2-theta'][1]=nominal
        for phase in phases:
            phase.data['Histograms'][hist.name]['Use']=True
        gpx.save()
    composition = mixture['composition']
    metadata = dict(schema='virtual-powder-forward/v1', configuration=config['id'],
        profile_sha256=sha256(profile), sources=sources, phases=phase_metadata,
        cell_mass_da=mixture['cell_mass_da'], cell_volume_a3=mixture['cell_volume_a3'], composition=composition,
        nuclear_cell_multiplier=cell_multiplier,
        intensity_basis='per primary reference-cell mass equivalent' if additions else 'original parent nuclear cell',
        mixture=bool(additions),phase_mass_fractions=fractions,volume_rule=mixture['volume_rule'],
        reference_parent_artifact='reference-parent.gpx',
        isotope_selection=mixture['isotope_selection'],isotope_options=mixture['isotope_options'],
        unit_scale=True, no_refinement=True, x_unit='degree_2theta' if cw else 'microsecond',
        gsas_scriptable_sha256=sha256(G.__file__), gsas_strmain_sha256=sha256(S.__file__))
    if cw:
        metadata['wavelength_a'] = float(inst['Lam'][1])
    if legacy_translation is not None:
        metadata['legacy_tof_profile_translation']=legacy_translation
    metadata['tof_profile_kernel']={'version':kernel_runs[0]['version'] if kernel_runs else
        ('native-CW-unchanged' if cw else 'installed-reference-unqualified'),
        'evaluations':kernel_runs,'experimental_accuracy_certified':False}
    if resolution_table is not None:metadata['resolution_table']=resolution_table.metadata()
    from gpx_recipe.live.validated_backend import active_identity
    metadata['calculation_backend'] = active_identity() or {'version': 'installed-unqualified'}
    if neutron_state:
        metadata['neutron_amplitude_model'] = {key:copy.deepcopy(value) for key,value in neutron_state.items()
                                               if key != 'native_selections'}
        metadata['reference_parent_isotope_basis'] = 'Native labels for symmetry; isotope mixtures use natural/default stock only in this parent artifact and declared atom fractions in the forward model'
    metadata['neutron_table'] = neutron_table
    metadata['cif_adp_normalizations'] = adp_normalizations
    metadata['resonant_angle_kernel'] = bool(extra)
    metadata['magnetic_moments'] = [{
        'phase':p.name, 'atom':a[0], 'components_muB':[float(v) for v in a[p.data['General']['AtomPtrs'][0]+4:p.data['General']['AtomPtrs'][0]+7]]}
        for p in phases if p.data['General']['Type']=='magnetic' for a in p.data['Atoms']]
    np.savez_compressed(output/'forward.npz', x=x, d=np.asarray(L.Pos2dsp(inst, x)), **curves, **extra)
    if not cw:
        metadata['tof_calibration'] = {key: float(inst.get(key, [0., 0.])[1])
                                       for key in ('difC', 'difA', 'difB', 'Zero', '2-theta')}
    (output/'forward.json').write_text(json.dumps(metadata, indent=2, default=lambda v:np.asarray(v).tolist())+'\n')
    return metadata, dict(x=x, d=np.asarray(L.Pos2dsp(inst, x)), **curves, **extra)
