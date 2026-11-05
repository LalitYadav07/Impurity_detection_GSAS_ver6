"""Versioned empirical minimum-distance policy from the user's MP catalog.

The minima are enforced as declared assumptions, not universal physical laws.
Histograms describe CrystalNN-selected bonds. They are not Gaussian priors on
all atom pairs, and observation counts are not independent-material counts.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from .chemical_geometry import ChemicalPotential,ContactRule,ChemistryUnqualified,NativeChemicalGeometry,ConditionedCubePrior,fingerprint


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def element_label(label):
    if label in ['D','T']:return 'H'
    match=re.fullmatch(r'([A-Z][a-z]?)(?:[+-][0-9]+|[0-9]+[+-]|_[0-9]+)?',str(label))
    if not match:raise ChemistryUnqualified('Unrecognized chemical species: '+str(label))
    return match.group(1)


class MaterialsProjectBondCatalog:
    def __init__(self,statistics,histogram,*,expected_statistics_sha256=None,expected_histogram_sha256=None):
        self.statistics=Path(statistics);self.histogram=Path(histogram)
        self.statistics_sha256=digest(statistics);self.histogram_sha256=digest(histogram)
        if ((expected_statistics_sha256 and expected_statistics_sha256!=self.statistics_sha256)
                or (expected_histogram_sha256 and expected_histogram_sha256!=self.histogram_sha256)):
            raise ValueError('Empirical chemistry source changed; a new protocol is required')
        self.pairs={};hist={}
        with self.statistics.open(newline='') as f:
            reader=csv.DictReader(f)
            if set(reader.fieldnames or [])!={'element_pair','bond_observations','minimum_angstrom','average_angstrom','stddev_angstrom'}:
                raise ValueError('Unrecognized bond-statistics schema')
            for row in reader:
                pair=tuple(row['element_pair'].split('-'))
                if len(pair)!=2 or tuple(sorted(map(element_label,pair)))!=pair or pair in self.pairs:
                    raise ValueError('Noncanonical or duplicate elemental pair')
                n=int(row['bond_observations']);minimum=float(row['minimum_angstrom']);mean=float(row['average_angstrom']);sd=float(row['stddev_angstrom'])
                if n<1 or not np.isfinite([minimum,mean,sd]).all() or minimum<=0 or mean<minimum or sd<0:
                    raise ValueError('Invalid bond statistics')
                self.pairs[pair]=dict(observations=n,minimum=minimum,mean=mean,stddev=sd)
        if not self.pairs:raise ValueError('Empty empirical chemistry catalog')
        with self.histogram.open(newline='') as f:
            reader=csv.DictReader(f)
            if set(reader.fieldnames or [])!={'element_pair','bin_start_angstrom','bin_end_angstrom','bond_observations'}:
                raise ValueError('Unrecognized bond histogram schema')
            for row in reader:
                pair=tuple(row['element_pair'].split('-'));lo=float(row['bin_start_angstrom']);hi=float(row['bin_end_angstrom']);n=int(row['bond_observations'])
                if pair not in self.pairs or not np.isfinite([lo,hi]).all() or lo<0 or hi<=lo or n<1:
                    raise ValueError('Invalid or unmatched bond histogram bin')
                hist.setdefault(pair,[]).append((lo,hi,n))
        if set(hist)!=set(self.pairs):raise ValueError('Statistics and histogram pair coverage differ')
        self.histograms={}
        for pair,bins in hist.items():
            b=np.array(sorted(bins),float);s=self.pairs[pair]
            if np.any(b[1:,0]<b[:-1,1]) or int(b[:,2].sum())!=s['observations']:
                raise ValueError('Histogram bins overlap or counts do not reconcile')
            if not b[0,0]<=s['minimum']<b[0,1]:raise ValueError('Recorded minimum is outside the first occupied histogram bin')
            b.flags.writeable=False;self.histograms[pair]=b
        self.fingerprint=fingerprint(self.manifest())

    def manifest(self):
        return dict(statistics_sha256=self.statistics_sha256,histogram_sha256=self.histogram_sha256,
            pairs=len(self.pairs),bond_observations=sum(v['observations'] for v in self.pairs.values()),
            source='User-provided Materials Project CrystalNN aggregate',units='angstrom',
            oxidation_states='merged into elemental pairs',counts='bond observations; not independent materials',
            distribution_scope='CrystalNN-assigned bonds in computed structures')

    def distribution(self,pair,distance):
        key=tuple(sorted(map(element_label,pair)))
        if key not in self.pairs:raise ChemistryUnqualified('No empirical observations for '+'-'.join(key))
        if not np.isfinite(distance) or distance<0:raise ValueError('Finite nonnegative bond length required')
        b=self.histograms[key];n=self.pairs[key]['observations']
        below=float(b[b[:,1]<=distance,2].sum());inside=(b[:,0]<=distance)&(b[:,1]>distance)
        uncertain=float(b[inside,2].sum())
        return dict(pair=key,distance_angstrom=float(distance),**self.pairs[key],
            empirical_cdf_interval=[below/n,(below+uncertain)/n],
            interpretation='Histogram-resolution bounds for assigned bonds, not probability that a structure is correct')

    def potential(self,species,*,partial_occupancy='unknown'):
        labels=tuple(map(element_label,species));types=sorted(set(labels));rules=[];missing=[]
        for i,a in enumerate(types):
            for b in types[i:]:
                key=(a,b)
                if key not in self.pairs:missing.append('-'.join(key));continue
                rules.append(ContactRule(key,'MP empirical observed minimum; user-declared policy; source '+self.statistics_sha256,
                                         hard_min=self.pairs[key]['minimum']))
        if missing:raise ChemistryUnqualified('Empirical contact policy has uncovered pairs: '+', '.join(missing))
        return ChemicalPotential(rules,partial_occupancy=partial_occupancy),labels


class EmpiricalNativeConstraint:
    """One policy shared by synthetic truth, priors, fitting and saved fits."""
    def __init__(self,native,catalog,*,partial_occupancy='unknown'):
        self.geometry=NativeChemicalGeometry(native.slow if hasattr(native,'slow') else native)
        self.catalog=catalog;self.potential,self.species=catalog.potential(self.geometry.species,partial_occupancy=partial_occupancy)
        self.fingerprint=fingerprint(dict(catalog=catalog.fingerprint,geometry=self.geometry.fingerprint,potential=self.potential.fingerprint,
            policy='all occupied periodic contacts must respect empirical minima',universal_physical_law=False))

    def assess(self,q,cell):
        result=self.potential.evaluate(*self.geometry.geometry(q,cell),self.species,self.geometry.occupancy)
        result['empirical_policy_fingerprint']=self.fingerprint
        result['universal_physical_law']=False
        for row in result['hard_violations']:
            row['distribution']=self.catalog.distribution(row['species'],row['distance'])
        return result

    def log_kernel(self,rows):
        rows=np.asarray(rows,float);nm=self.geometry.nm;nc=self.geometry.nc
        if rows.ndim!=2 or rows.shape[1]<nm+nc:raise ValueError('Native geometry parameter columns required')
        return np.array([self.assess(r[:nm],r[nm:nm+nc])['log_kernel'] for r in rows])

    def cube_prior(self,bounds):
        return ConditionedCubePrior(bounds,self.log_kernel,context_fingerprint=self.fingerprint)

    def require(self,q,cell):
        result=self.assess(q,cell)
        if not result['feasible']:
            raise EmpiricalConfigurationRejected(result)
        return result


class EmpiricalConfigurationRejected(ValueError):
    def __init__(self,result):
        self.assessment=result
        super().__init__('Configuration violates the declared empirical bond minima; this is not subgroup exclusion')


class EmpiricalCheckedForward:
    """Check before native forward evaluation; priors must also be conditioned.

    Numerical derivative probes are checked explicitly. This guard must not be
    substituted for the normalized conditioned prior when calculating evidence.
    """
    def __init__(self,native,constraint):self.native=native;self.constraint=constraint

    def evaluate(self,theta,*,jacobian=False,cell_step=1e-4):
        theta=np.asarray(theta,float);nm=self.constraint.geometry.nm;nc=self.constraint.geometry.nc
        self.constraint.require(theta[:nm],theta[nm:nm+nc])
        if jacobian:
            for j in range(nm,nm+nc):
                for sign in [-1,1]:
                    probe=theta.copy();probe[j]+=sign*cell_step
                    self.constraint.require(probe[:nm],probe[nm:nm+nc])
        return self.native.evaluate(theta,jacobian=jacobian,cell_step=cell_step)


class EmpiricalCheckedSlowForward:
    """Guard independent saved-GPX calculations and export, too.

    The wrapped geometry is a fixed prepared candidate. Reanchoring, changing
    symmetry or replacing a phase requires a new constraint and wrapper.
    """
    def __init__(self,native,constraint):
        self.native=native;self.constraint=constraint;self.last_geometry=None
        self.definition_key=self._definition_key()

    def __getattr__(self,name):return getattr(self.native,name)

    def _definition_key(self):
        phase=self.native.phase.data;cx,ct,_,_=phase['General']['AtomPtrs']
        sg=phase['General']['SGData'];iso=phase.get('ISODISTORT',{})
        return fingerprint(dict(species=[str(a[ct]) for a in phase['Atoms']],
            occupancy=[float(a[cx+3]) for a in phase['Atoms']],
            symmetry=dict(inversion=bool(sg.get('SGInv')),fixed=bool(sg.get('SGFixed')),
                centers=np.asarray(sg['SGCen']).tolist(),
                operations=[(np.asarray(r).tolist(),np.asarray(t).tolist()) for r,t in sg['SGOps']]),
            mode_origins={k:np.asarray(iso.get(k,[])).tolist() for k in ['G2parentCoords','G2coordOffset']},
            basis={k:np.asarray(v).tolist() for k,v in self.native.basis.items()}))

    def _require_definition(self):
        if self._definition_key()!=self.definition_key:
            raise ValueError('Native chemical geometry definition changed; rebuild the constraint')

    def evaluate(self,q,cell=None,**kwargs):
        self._require_definition()
        q=np.asarray(q,float)
        if cell is None:cell=np.zeros(self.constraint.geometry.nc)
        cell=np.asarray(cell,float)
        self.constraint.require(q,cell)
        result=self.native.evaluate(q,cell,**kwargs)
        self.last_geometry=(q.copy(),cell.copy())
        return result

    def save_verified_prediction(self,prediction):
        self._require_definition()
        if self.last_geometry is None:
            raise ValueError('Export requires a chemistry-checked native evaluation first')
        q,cell=self.last_geometry;g=self.constraint.geometry
        self.constraint.require(q,cell)
        phase=self.native.phase.data;cx=phase['General']['AtomPtrs'][0]
        actual=np.asarray([atom[cx:cx+3] for atom in phase['Atoms']],float)
        expected=g.asymmetric_base+g.asymmetric_mapping@q
        if actual.shape!=expected.shape or not np.allclose(actual,expected,rtol=0,atol=1e-10):
            raise ValueError('Native atom coordinates changed after the chemistry-checked evaluation')
        from gpx_recipe.symmetry.displacements import metric
        expected_metric=g.geometry(q,cell)[1]
        if not np.allclose(metric(phase['General']['Cell'][1:7]),expected_metric,rtol=1e-10,atol=1e-10):
            raise ValueError('Native cell changed after the chemistry-checked evaluation')
        return self.native.save_verified_prediction(prediction)


def load_declared_catalog(policy_path):
    path=Path(policy_path)
    if not path.is_file():raise ChemistryUnqualified('Required empirical chemistry policy is missing: '+str(path))
    policy=json.loads(path.read_text())
    if (not isinstance(policy,dict) or type(policy.get('schema_version')) is not int
            or policy['schema_version']!=1):
        raise ChemistryUnqualified('Unsupported empirical chemistry policy schema')
    if policy.get('mode')!='hard_empirical_minimum':raise ChemistryUnqualified('Unsupported empirical chemistry enforcement mode')
    # The low-level catalog permits unhashed exploratory input. A declared
    # calculation policy must never opt out of source pinning via null/empty
    # hashes: a resumed calculation would then silently use different chemistry.
    for name in ['statistics','histogram']:
        expected=policy.get(name+'_sha256')
        if not isinstance(expected,str) or re.fullmatch(r'[0-9a-f]{64}',expected) is None:
            raise ChemistryUnqualified('Required pinned SHA-256 is invalid: '+name)
        if not isinstance(policy.get(name),str) or not policy[name].strip():
            raise ChemistryUnqualified('Required empirical chemistry source path is invalid: '+name)
    def source(name):
        value=Path(policy[name]);return value if value.is_absolute() else path.parent/value
    return MaterialsProjectBondCatalog(source('statistics'),source('histogram'),
        expected_statistics_sha256=policy['statistics_sha256'],expected_histogram_sha256=policy['histogram_sha256'])
