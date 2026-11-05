"""Explicit chemistry potentials on periodic structures, not a scattering model.

No element-dependent cutoff is invented here. Rules are reviewed inputs. Pair
rules are lower-contact bounds, never upper bounds on *every* pair of elements.
The soft penalty is per occupied atom so repeating a unit cell does not repeat
the same chemical evidence. Partially occupied contacts require a declared
joint-occupancy model; unknown correlation is an unresolved calculation.
"""
from dataclasses import dataclass, asdict
from itertools import product
import hashlib
import json
import numpy as np

from .physical_prior_geometry import metric_matrix, native_reciprocal_matrix


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


class ChemistryUnqualified(ValueError):
    """Missing chemical/occupancy context cannot become model exclusion."""


def direct_metric_from_reciprocal(reciprocal):
    """Invert an SPD metric without introducing an asymmetric roundoff part.

    General LU inversion may put opposite roundoff signs into nominally zero
    off-diagonal entries. L^-T L^-1 preserves symmetry by construction, rather
    than broadening validation tolerances on caller-supplied metrics.
    """
    g=metric_matrix(reciprocal)
    inverse_l=np.linalg.solve(np.linalg.cholesky(g),np.eye(3))
    return metric_matrix(inverse_l.T@inverse_l)


@dataclass(frozen=True)
class ContactRule:
    species: tuple
    source: str
    hard_min: float = 0.
    soft_min: float = 0.
    soft_sigma: float = 1.
    conditions: tuple = ()  # (name, lower, upper), inclusive; e.g. pressure_GPa

    def __post_init__(self):
        if (len(self.species)!=2 or any(not isinstance(s,str) or not s for s in self.species)
                or not isinstance(self.source,str) or not self.source.strip()
                or not np.isfinite([self.hard_min,self.soft_min,self.soft_sigma]).all()
                or self.hard_min<0 or self.soft_min<0 or max(self.hard_min,self.soft_min)<=0 or self.soft_sigma<=0):
            raise ValueError('Explicit species, provenance and finite positive contact scale required')
        names=[]
        for item in self.conditions:
            if len(item)!=3:raise ValueError('Condition must name its inclusive interval')
            name,low,high=item
            if not name or not np.isfinite([low,high]).all() or low>high:raise ValueError('Invalid chemistry applicability interval')
            names.append(name)
        if len(names)!=len(set(names)):raise ValueError('Duplicate chemical applicability condition')
        object.__setattr__(self,'species',tuple(sorted(self.species)))
        object.__setattr__(self,'conditions',tuple(tuple(x) for x in self.conditions))

    def applies(self,context):
        return all(name in context and np.isfinite(context[name]) and low<=context[name]<=high for name,low,high in self.conditions)


def periodic_contacts(coordinates,direct_metric,cutoff,*,max_images=2000000):
    """Every unordered periodic contact inside cutoff, each counted once.

    Reciprocal-metric Cauchy bounds enumerate all relevant integer images in
    arbitrary skew cells. Rounding fractional deltas is NOT a closest-image
    algorithm. Distinct site indices at coincident positions are retained.
    Coordinates can be unwrapped; returned translations use those coordinates.
    """
    xyz=np.asarray(coordinates,float);g=metric_matrix(direct_metric)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or not np.isfinite(xyz).all():raise ValueError('Finite site coordinates required')
    if not np.isfinite(cutoff) or cutoff<=0 or type(max_images) is not int or max_images<1:raise ValueError('Bounded positive contact search required')
    inverse=np.linalg.inv(g);radius=cutoff*np.sqrt(np.diag(inverse))
    examined=0;records=[];roundoff=64*np.finfo(float).eps
    for i in range(len(xyz)):
        for j in range(i,len(xyz)):
            delta=xyz[j]-xyz[i]
            lo=np.ceil(-delta-radius-roundoff*(abs(delta)+radius+1)).astype(int)
            hi=np.floor(-delta+radius+roundoff*(abs(delta)+radius+1)).astype(int)
            count=int(np.prod(np.maximum(hi-lo+1,0),dtype=object));examined+=count
            if examined>max_images:raise ChemistryUnqualified('Periodic-image budget exceeded; no feasibility conclusion')
            if not count:continue
            shifts=np.array(list(product(*(range(int(a),int(b)+1) for a,b in zip(lo,hi)))),dtype=int)
            if i==j:
                # Keep only the lexicographically positive half, excluding 0.
                keep=(shifts[:,0]>0)|((shifts[:,0]==0)&(shifts[:,1]>0))|((shifts[:,0]==0)&(shifts[:,1]==0)&(shifts[:,2]>0))
                shifts=shifts[keep]
            v=delta+shifts;r2=np.einsum('ni,ij,nj->n',v,g,v)
            magnitude=np.einsum('ni,ij,nj->n',abs(v),abs(g),abs(v))
            tolerance=roundoff*np.maximum(magnitude,cutoff**2)
            if np.any(r2 < -tolerance):raise ValueError('Negative distance from unstable cell metric')
            for shift,value in zip(shifts[r2<=cutoff**2+tolerance],r2[r2<=cutoff**2+tolerance]):
                records.append((i,j,tuple(map(int,shift)),float(np.sqrt(max(value,0)))))
    return records


def hard_contact_comparison(distance,minimum,first,second,shift,direct_metric):
    """Compare a hard contact with an explicit floating-point error bound.

    This is an arithmetic allowance, never a physical bond-length tolerance.
    Large cancellation uncertainty is unresolved rather than accepted.
    """
    first=np.asarray(first,float);second=np.asarray(second,float);shift=np.asarray(shift,float)
    v=second-first+shift;g=np.abs(direct_metric);eps=np.finfo(float).eps
    coordinate_error=4*eps*(abs(first)+abs(second)+abs(shift))
    quadratic_error=64*eps*max(float(abs(v)@g@abs(v)),distance**2,minimum**2)
    quadratic_error+=float(2*abs(v)@g@coordinate_error+coordinate_error@g@coordinate_error)
    allowance=quadratic_error/max(distance+minimum,np.finfo(float).tiny)
    if allowance>1e-8*max(minimum,distance,np.finfo(float).tiny):
        raise ChemistryUnqualified('Contact precision is insufficient for the declared hard bound')
    gap=distance-minimum
    return dict(violates=bool(gap < -allowance),roundoff_boundary=bool(-allowance<=gap<0),
                distance_minus_minimum=gap,arithmetic_allowance_angstrom=allowance)


class ChemicalPotential:
    def __init__(self,rules,*,context=None,partial_occupancy='unknown',max_images=2000000):
        if not rules or any(not isinstance(r,ContactRule) for r in rules):raise ValueError('Nonempty reviewed contact-rule list required')
        self.rules=tuple(rules);self.context=dict(context or {});self.partial_occupancy=partial_occupancy;self.max_images=max_images
        if partial_occupancy not in ['unknown','independent_static_sites']:raise ValueError('Explicit occupancy correlation policy required')
        if len({r.species for r in rules})!=len(rules):raise ValueError('One combined rule per species pair required')
        if any(not r.applies(self.context) for r in rules):raise ChemistryUnqualified('Chemical rule is outside its declared applicability or context is missing')
        self.by_pair={r.species:r for r in rules}
        self.fingerprint=fingerprint(self.manifest())

    def manifest(self):
        return dict(rules=[asdict(r) for r in self.rules],context=self.context,partial_occupancy=self.partial_occupancy,
            penalty_normalization='per_occupied_atom',max_images=self.max_images,
            hard_bound_comparison='squared_distance_arithmetic_bound_v1; unresolved precision fails closed',
            scope='declared lower-contact rules; no implicit valence or bond topology assumption')

    def evaluate(self,coordinates,direct_metric,species,occupancy,*,joint_occupancy=None):
        xyz=np.asarray(coordinates,float);occ=np.asarray(occupancy,float);species=tuple(species)
        if (occ.shape!=(len(xyz),) or len(species)!=len(xyz) or not np.isfinite(occ).all()
                or np.any(occ<0) or np.any(occ>1) or occ.sum()<=0 or any(not s for s in species)):
            raise ValueError('Per-site species and occupancy in [0,1] required')
        # Explicit per-image joint probabilities can describe exclusive split
        # alternatives. Keys follow the canonical orientation returned below.
        joint={} if joint_occupancy is None else dict(joint_occupancy)
        for key,value in joint.items():
            if (not isinstance(key,tuple) or len(key)!=3 or not isinstance(key[2],tuple) or len(key[2])!=3
                    or any(type(n) is not int for n in (key[0],key[1],*key[2]))
                    or not 0<=key[0]<=key[1]<len(xyz) or not np.isfinite(value)):
                raise ValueError('Canonical finite per-image joint occupancy required')
            i,j,_=key
            if not max(0.,occ[i]+occ[j]-1.)-1e-12<=value<=min(occ[i],occ[j])+1e-12:
                raise ValueError('Joint occupancy violates marginal probability bounds')
        cutoff=max(max(r.hard_min,r.soft_min) for r in self.rules)
        contacts=periodic_contacts(xyz,direct_metric,cutoff,max_images=self.max_images)
        penalty=0.;violations=[];uncertain=[];active=[];roundoff_contacts=[]
        for i,j,shift,d in contacts:
            rule=self.by_pair.get(tuple(sorted((species[i],species[j]))))
            if rule is None or min(occ[i],occ[j])==0:continue
            if d>=max(rule.hard_min,rule.soft_min):continue
            key=(i,j,shift)
            if key in joint:probability=joint[key]
            elif occ[i]==1. and occ[j]==1.:probability=1.
            elif self.partial_occupancy=='independent_static_sites':probability=occ[i]*occ[j]
            else:uncertain.append(dict(i=i,j=j,image=shift,distance=d));continue
            if probability==0:continue
            row=dict(i=i,j=j,image=shift,distance=d,species=rule.species,joint_occupancy=probability,source=rule.source)
            if d<rule.hard_min:
                comparison=hard_contact_comparison(d,rule.hard_min,xyz[i],xyz[j],shift,direct_metric)
                if comparison['violates']:violations.append(dict(row,hard_min=rule.hard_min))
                elif comparison['roundoff_boundary']:roundoff_contacts.append(dict(row,hard_min=rule.hard_min,comparison=comparison))
            amount=probability*(max(rule.soft_min-d,0.)/rule.soft_sigma)**2
            penalty+=amount;active.append(dict(row,soft_penalty=amount))
        if uncertain:raise ChemistryUnqualified('Partial-occupancy contacts need an explicit joint model: '+str(uncertain[:3]))
        penalty/=float(occ.sum())
        return dict(feasible=not violations,soft_penalty=penalty,log_kernel=-.5*penalty if not violations else -np.inf,
            hard_violations=violations,active_contacts=active,roundoff_boundary_contacts=roundoff_contacts,
            periodic_contacts_examined=len(contacts),
            potential_fingerprint=self.fingerprint,scope=self.manifest()['scope'])


class ConditionedCubePrior:
    """Exact rejection draws from uniform-cube base × exp(log_potential).

    The callback must be deterministic and return values <= 0 or -inf. The
    normalized prior is defined by accepted sampling, so SMC evidence starts
    as an expectation under that conditional prior. Model priors are unchanged;
    retained chemistry mass is NOT silently used as extra model evidence.
    MH uses log_kernel differences; its unknown normalizer cancels. Normalizer
    estimation for external-chemistry model evidence is a separate operation.
    """
    def __init__(self,bounds,log_potential,*,context_fingerprint,max_attempts=1000000):
        self.bounds=np.array(bounds,float,copy=True)
        if (self.bounds.ndim!=2 or self.bounds.shape[1]!=2 or not len(self.bounds) or not np.isfinite(self.bounds).all()
                or np.any(np.diff(self.bounds,axis=1)<=0) or not context_fingerprint
                or type(max_attempts) is not int or max_attempts<1):raise ValueError('Finite bounds and chemistry identity required')
        self.bounds.flags.writeable=False;self.log_potential=log_potential;self.max_attempts=max_attempts
        self.fingerprint=fingerprint(dict(bounds=self.bounds.tolist(),context=context_fingerprint,policy='chemistry_conditioned_model_priors_unchanged'))
        self.last_sampling=None

    def log_kernel(self,rows):
        rows=np.asarray(rows,float)
        if rows.ndim!=2 or rows.shape[1]!=len(self.bounds) or not np.isfinite(rows).all():raise ValueError('Finite parameter rows required')
        inside=np.all((rows>=self.bounds[:,0])&(rows<=self.bounds[:,1]),axis=1);out=np.full(len(rows),-np.inf)
        if inside.any():
            values=np.asarray(self.log_potential(rows[inside]),float)
            if values.shape!=(int(inside.sum()),) or np.isnan(values).any() or np.any(values>0):raise ValueError('Chemistry log-potential must be <=0 with -inf for infeasible states')
            out[inside]=values
        return out

    def sample(self,rng,count):
        if type(count) is not int or count<1:raise ValueError('Positive prior sample count required')
        accepted=[];attempted=0;found=0
        while found<count and attempted<self.max_attempts:
            batch=min(max(count-found,32),self.max_attempts-attempted)
            values=rng.uniform(*self.bounds.T,size=(batch,len(self.bounds)));kernels=self.log_kernel(values)
            keep=np.log(rng.random(batch))<kernels;accepted.append(values[keep]);found+=int(keep.sum());attempted+=batch
        self.last_sampling=dict(attempted=attempted,accepted=found,requested=count,
            policy='exact_rejection_from_conditional_prior',acceptance_fraction_is_not_model_probability=True)
        if found<count:raise ChemistryUnqualified('Chemistry prior rejection sampler exhausted budget; no model exclusion')
        return np.concatenate(accepted)[:count]


class NativeChemicalGeometry:
    """Candidate-local geometry lifted from checked native mode coordinates."""
    def __init__(self,native_candidate):
        n=native_candidate;phase=n.phase.data;cx,ct,cs,_=phase['General']['AtomPtrs'];sg=phase['General']['SGData']
        # Native screw translations are often float32 (e.g. 1/3). Equivalent
        # operations can differ by several storage ulps after composition.
        # Bound that representation error from the actual native dtype/values;
        # never use GenAtom's broad 2e-4 positional merging tolerance.
        precision=64*np.finfo(float).eps
        for value in [sg['SGCen'],*[t for _,t in sg['SGOps']]]:
            value=np.asarray(value)
            if np.issubdtype(value.dtype,np.floating):
                precision=max(precision,float(8*np.max(abs(np.spacing(value)))))
        self.operation_roundoff=precision;self.maximum_merged_roundoff=0.
        self.nm=n.basis['physical_basis'].shape[1];self.nc=n.basis['cell_basis'].shape[1]
        base=np.array([a[cx:cx+3] for a in phase['Atoms']],float)
        mapping=np.einsum('ain,nm->aim',n.basis['coordinate_basis'],n.basis['physical_basis'])
        # Preparation may anchor a printed child coordinate to a parent orbit
        # carrying float32 translation error. Use the recorded pre-anchor
        # stabilizer only when its allowed motion is unchanged, and bound the
        # resulting discrepancy by that exact recorded coordinate shift.
        anchors={r['child']:r for r in getattr(n,'meta',{}).get('exact_parent_anchor',{}).get('sites',[])}
        expanded=[];maps=[];species=[];occ=[];orbit=[];operators=[]
        for i,atom in enumerate(phase['Atoms']):
            positions=[];directions=[];local_ops=[];before_positions=[]
            anchor=anchors.get(str(atom[ct-1]));before=base[i];anchor_shift=0.
            if anchor is not None:
                after=np.asarray(anchor['after'],float)
                if after.shape!=(3,) or not np.allclose(after,base[i],rtol=0,atol=1e-12):
                    raise ChemistryUnqualified('Recorded parent anchor no longer matches native zero-mode coordinates')
                before=np.asarray(anchor['before'],float)
                anchor_shift=float(np.max(abs(after-before)))
            for center in sg['SGCen']:
                for sign in ([1,-1] if sg['SGInv'] and not sg.get('SGFixed',False) else [1]):
                    for rot,translation in sg['SGOps']:
                        rotation=sign*np.asarray(rot,float);offset=sign*np.asarray(translation,float)+center
                        pos=rotation@base[i]+offset;direction=rotation@mapping[i];before_pos=rotation@before+offset
                        equivalent=False
                        for old,oldmap,old_before,(old_rotation,_) in zip(positions,directions,before_positions,local_ops):
                            delta=pos-old
                            difference=float(np.max(abs(delta-np.rint(delta))))
                            delta_before=before_pos-old_before
                            before_difference=float(np.max(abs(delta_before-np.rint(delta_before))))
                            anchor_bound=float(np.linalg.norm(rotation-old_rotation,ord=np.inf))*anchor_shift+precision
                            same_position=(difference<=precision or (before_difference<=precision and difference<=anchor_bound))
                            if same_position and np.allclose(direction,oldmap,rtol=1e-10,atol=1e-10):
                                self.maximum_merged_roundoff=max(self.maximum_merged_roundoff,difference)
                                equivalent=True;break
                        if not equivalent:
                            positions.append(pos);directions.append(direction);local_ops.append((rotation,offset));before_positions.append(before_pos)
            if len(positions)!=int(round(float(atom[cs+1]))):
                raise ChemistryUnqualified('Native orbit multiplicity disagrees with stable geometry expansion for '+str(atom[ct-1])+f': expanded={len(positions)}, native={atom[cs+1]}, operation_roundoff={precision}')
            expanded.extend(positions);maps.extend(directions);species.extend([str(atom[ct])]*len(positions))
            occ.extend([float(atom[cx+3])]*len(positions));orbit.extend([i]*len(positions));operators.extend(local_ops)
        self.base=np.array(expanded);self.mapping=np.array(maps);self.species=tuple(species);self.occupancy=np.array(occ)
        self.asymmetric_base=base;self.asymmetric_mapping=mapping;self.orbit=np.array(orbit);self.operators=operators
        self.a0=n.basis['baseline_a'].copy();self.cell_mapping=n.a_scale*n.basis['cell_basis']
        self.fingerprint=fingerprint(dict(base=self.base.tolist(),mapping=self.mapping.tolist(),species=self.species,
            occupancy=self.occupancy.tolist(),a0=self.a0.tolist(),cell_mapping=self.cell_mapping.tolist(),operation_roundoff=precision))

    def geometry(self,q,cell):
        q=np.asarray(q,float);cell=np.asarray(cell,float)
        if q.shape!=(self.nm,) or cell.shape!=(self.nc,) or not np.isfinite(q).all() or not np.isfinite(cell).all():raise ValueError('Finite native geometry coordinates required')
        reciprocal=metric_matrix(native_reciprocal_matrix(self.a0+self.cell_mapping@cell))
        return self.base+self.mapping@q,direct_metric_from_reciprocal(reciprocal)
