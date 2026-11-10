"""Explicit-CIF nuclear lines for the physical elastic count operator.

Every stored reciprocal point represents its Friedel pair, with multiplicity
two. Atomic positions, occupations and anisotropic displacement tensors are
expanded by the independent CIF reader. Complex isotope amplitudes are applied
at each actual scattering wavelength; the two Friedel intensities are averaged
before multiplicity is applied. No crystal symmetry grouping or intensity floor
is used. This does not certify the CIF/specimen identity or atomic input data.
"""
from dataclasses import dataclass
from pathlib import Path
import copy,hashlib,json,math
import numpy as np
from .nuclear import full_cell
from .material import _metric,_partial,atom_type
from .isotopes import amplitude

VERSION='powgen-explicit-cif-nuclear-lines/1'


def _records_hash(types,records):
    return hashlib.sha256(json.dumps(dict(zip(types,records)),sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class NuclearLines:
    hkl: np.ndarray
    d_a: np.ndarray
    multiplicity: np.ndarray
    atom_types: tuple
    neutron_records: tuple
    partial_amplitudes: np.ndarray
    cell_volume_a3: float
    cell_mass_da: float
    atom_count: float
    minimum_wavelength_a: float
    provenance: dict

    def factor_squared_b(self,indices,wavelength_a):
        """Friedel-averaged squared amplitude, barn per declared nuclear cell."""
        i=np.asarray(indices);w=np.asarray(wavelength_a,float)
        if (i.ndim!=1 or i.dtype.kind not in 'iu' or len(i)>100000 or w.shape!=i.shape
                or np.any(i<0) or np.any(i>=len(self.d_a)) or not np.isfinite(w).all()
                or np.any(w<self.minimum_wavelength_a)):
            raise ValueError('Bounded valid reflection indices and covered physical wavelengths are required')
        if _records_hash(self.atom_types,self.neutron_records)!=self.provenance['neutron_records_sha256']:
            raise ValueError('Compiled neutron records changed; rebuild the nuclear line model')
        b=np.array([amplitude(r,w) for r in self.neutron_records]).T
        partial=self.partial_amplitudes[i]
        forward=np.einsum('ne,ne->n',partial,b)
        reverse=np.einsum('ne,ne->n',partial.conj(),b)
        result=(abs(forward)**2+abs(reverse)**2)/2
        if not np.isfinite(result).all():
            raise ValueError('Nuclear amplitudes exceed their finite squared-intensity domain')
        return result

    def cross_section(self,wavelength_a,*,max_factor_values=2000000):
        """Total supplied-domain coherent elastic removal, barn per nuclear cell."""
        from .elastic_powder import powder_cross_section
        w=np.asarray(wavelength_a,float)
        if (w.ndim!=1 or not 1<=len(w)<=100000 or not np.isfinite(w).all()
                or np.any(w<self.minimum_wavelength_a)):
            raise ValueError('Wavelengths must be finite vectors inside the compiled reflection domain')
        if not len(self.d_a):
            return np.zeros_like(w)
        return powder_cross_section(self.d_a,self.multiplicity,self.cell_volume_a3,
            self.factor_squared_b,w,max_factor_values=max_factor_values)


def nuclear_lines(path,neutron_records,minimum_wavelength_a,*,max_reflections=250000,max_bytes=256*1024**2):
    """Compile complete Friedel pairs with 2*d >= minimum incident wavelength.

    The component retains complex site sums instead of differences of large
    cumulative Gram matrices, so weak individual reflections do not lose their
    amplitudes through cumulative subtraction. Exact charged atom types and
    isotope-weighted masses remain explicit. Resources are checked before the
    reciprocal cube or site-sum work arrays are allocated.
    """
    path=Path(path).resolve()
    if (type(minimum_wavelength_a) not in (int,float) or not math.isfinite(minimum_wavelength_a)
            or minimum_wavelength_a<=0 or type(max_reflections) is not int
            or not 1<=max_reflections<=1000000 or type(max_bytes) is not int
            or not 1024<=max_bytes<=1024**3):
        raise ValueError('A positive wavelength domain and bounded reciprocal resources are required')
    xyz,tensors,raw_species,occupation,counts=full_cell(path,exact_atom_types=True)
    species=np.array([atom_type(v) for v in raw_species]);types=tuple(sorted(set(species)))
    if not isinstance(neutron_records,dict) or not 1<=len(types)<=16 or set(types)-set(neutron_records):
        raise ValueError('Explicit isotope records for every exact charged CIF atom type are required')
    records=tuple(copy.deepcopy(neutron_records[k]) for k in types)
    for record in records:
        if not isinstance(record,dict) or 'Mass' not in record:
            raise ValueError('Positive isotope-weighted atomic masses are required')
        mass=record['Mass']
        if type(mass) not in (int,float) or not math.isfinite(mass) or mass<=0:
            raise ValueError('Positive isotope-weighted atomic masses are required')
        amplitude(record,np.array([minimum_wavelength_a]))
    lengths,reciprocal,volume=_metric(path)
    ratios=lengths/(minimum_wavelength_a/2)
    if not np.isfinite(ratios).all() or np.any(ratios>1000000):
        raise ValueError('Nuclear reciprocal enumeration exceeds its resource budget')
    bounds=np.floor(ratios).astype(np.int32)
    cube=math.prod(int(2*v+1) for v in bounds)
    if cube>2000000 or cube*48>max_bytes:
        raise ValueError('Nuclear reciprocal enumeration exceeds its resource budget')
    candidates=np.indices(tuple(2*bounds+1),dtype=np.int32).reshape(3,-1).T-bounds
    h,k,l=candidates.T
    half=(h>0)|((h==0)&(k>0))|((h==0)&(k==0)&(l>0))
    hkl=candidates[half]
    d=1/np.sqrt(np.einsum('hi,ij,hj->h',hkl,reciprocal,hkl))
    chosen=d>=minimum_wavelength_a/2
    hkl,d=hkl[chosen],d[chosen]
    if len(d)>max_reflections:
        raise ValueError('Nuclear reflection count exceeds its resource budget')
    required=len(d)*(64+16*len(types))+len(xyz)*256
    if required>max_bytes:
        raise ValueError('Retained nuclear line arrays exceed their resource budget')
    order=np.argsort(-d);hkl,d=hkl[order],d[order]
    del candidates,h,k,l,half,chosen,order
    chunk=min(1024,(max_bytes-required)//(80*len(xyz)+48*len(types)))
    if chunk<1:
        raise ValueError('Nuclear site-sum work exceeds its resource budget')
    partial=np.empty((len(d),len(types)),complex)
    for first in range(0,len(d),chunk):
        last=min(first+chunk,len(d))
        partial[first:last]=_partial(hkl[first:last],xyz,tensors,occupation,species,types)
    multiplicity=np.full(len(d),2,dtype=np.int64)
    cell_mass=float(sum(occupation[species==kind].sum()*record['Mass'] for kind,record in zip(types,records)))
    if cell_mass<=0:
        raise ValueError('At least one occupied nuclear site is required')
    for array in (hkl,d,partial,multiplicity):array.setflags(write=False)
    provenance=dict(version=VERSION,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        cif_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        dependency_sources={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
            for n in ['nuclear.py','material.py','isotopes.py']},
        neutron_records_sha256=_records_hash(types,records),minimum_wavelength_a=minimum_wavelength_a,
        full_reciprocal_rows=2*len(d),friedel_half_rows=len(d),independent_site_multiplicities=counts,
        all_displacement_tensors_preserved=True,intensity_floor_or_symmetry_grouping_added=False,
        specimen_identity_or_metrology_verified=False,experimental_accuracy_certified=False)
    return NuclearLines(hkl,d,multiplicity,types,records,partial,volume,cell_mass,
        float(occupation.sum()),float(minimum_wavelength_a),provenance)
