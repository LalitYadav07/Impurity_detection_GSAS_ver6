"""Candidate-local hard-contact calculations over a bounded linear domain.

Triangle and relative-metric spectral bounds safely discard contacts that
cannot violate a declared minimum anywhere in the domain. Retained contacts
use the full, current cell metric. No scattering profile is approximated.
"""
from itertools import product
import numpy as np
from .chemical_geometry import ChemistryUnqualified,fingerprint,direct_metric_from_reciprocal
from .physical_prior_geometry import native_reciprocal_matrix,metric_matrix


def geometry_identity(base,mapping,baseline_a,cell_mapping,species,occupancy):
    from .materials_project_chemistry import element_label
    return fingerprint(dict(base=np.asarray(base).tolist(),mapping=np.asarray(mapping).tolist(),
        baseline_a=np.asarray(baseline_a).tolist(),cell_mapping=np.asarray(cell_mapping).tolist(),
        species=list(map(element_label,species)),occupancy=np.asarray(occupancy).tolist()))


class BoundedChemicalGeometry:
    def __init__(self,base,mapping,baseline_a,cell_mapping,species,occupancy,lift,dimensions,radii,catalog,*,max_images=2000000):
        base=np.asarray(base,float);mapping=np.asarray(mapping,float);cell_mapping=np.asarray(cell_mapping,float)
        lift=np.asarray(lift,float);occupancy=np.asarray(occupancy,float)
        self.dimensions=tuple(dimensions);self.radii=tuple(float(v) for v in radii);self.offsets=np.r_[0,np.cumsum(dimensions)]
        self.dimension=sum(dimensions);nm=mapping.shape[-1] if mapping.ndim==3 else -1
        nc=cell_mapping.shape[-1] if cell_mapping.ndim==2 else -1
        if (base.ndim!=2 or base.shape[1]!=3 or mapping.shape!=(len(base),3,nm) or cell_mapping.shape!=(6,nc)
                or lift.shape!=(nm+nc,self.dimension) or len(species)!=len(base) or occupancy.shape!=(len(base),)
                or any(type(d)is not int or d<0 for d in dimensions) or self.dimension<1 or len(radii)!=len(dimensions)
                or not np.isfinite(self.radii).all() or min(self.radii)<=0
                or not all(np.isfinite(x).all() for x in [base,mapping,cell_mapping,lift,occupancy])):
            raise ValueError('Finite native geometry and a bounded linear parameter domain required')
        if not np.all(occupancy==1.):
            raise ChemistryUnqualified('Bounded contact prior requires full static occupancy; joint occupancy is not inferred')
        potential,labels=catalog.potential(species)
        self.geometry_identity=geometry_identity(base,mapping,baseline_a,cell_mapping,labels,occupancy)
        self.catalog_fingerprint=catalog.fingerprint;self.lift=lift.copy();self.lift.flags.writeable=False
        self.baseline=metric_matrix(native_reciprocal_matrix(baseline_a))
        conditioning=float(np.linalg.cond(self.baseline))
        if conditioning>1e10:raise ChemistryUnqualified('Reference metric is too ill-conditioned for bounded contact certification')
        arithmetic=256*np.finfo(float).eps*max(1.,conditioning)
        effective_radii=[r*(1+128*np.finfo(float).eps*max(n,1)) for n,r in zip(dimensions,radii)]
        direct=direct_metric_from_reciprocal(self.baseline);cartesian=np.linalg.cholesky(direct).T
        cell_lift=cell_mapping@lift[nm:]
        self.cell_directions=np.array([native_reciprocal_matrix(v) for v in cell_lift.T])
        mode_mapping=np.einsum('aim,mk->aik',mapping,lift[:nm])
        inverse_l=np.linalg.solve(np.linalg.cholesky(self.baseline),np.eye(3))
        relative=np.array([inverse_l@v@inverse_l.T for v in self.cell_directions])
        # Frobenius norm bounds spectral norm, hence
        # Gdirect(z) >= Gdirect(0)/(1+relative_bound) in Loewner order.
        metric_bound=sum(r*np.linalg.norm(relative[a:b].reshape(b-a,9).T,ord=2) if b>a else 0.
                         for a,b,r in zip(self.offsets[:-1],self.offsets[1:],effective_radii))
        self.metric_bound=float(metric_bound*(1+arithmetic)+arithmetic)
        if self.metric_bound>=1:raise ChemistryUnqualified('Declared cell domain has no certified positive metric bound')
        retained=[];examined=0;safe=0;impossible=[];eps=arithmetic
        for i in range(len(base)):
            for j in range(i,len(base)):
                rule=potential.by_pair[tuple(sorted((labels[i],labels[j])))];threshold=rule.hard_min
                if threshold == 0:
                    # A source-screened pair can intentionally have no bound.
                    # Do not enumerate images or construct optimizer contacts
                    # for that absent inequality.
                    continue
                delta=base[j]-base[i];directions=mode_mapping[j]-mode_mapping[i]
                displacement=sum(r*np.linalg.norm(cartesian@directions[:,a:b],ord=2) if b>a else 0.
                                 for a,b,r in zip(self.offsets[:-1],self.offsets[1:],effective_radii))
                # Inflate rather than clip a physical parameter range. The
                # only allowance is tied to floating-point arithmetic size.
                displacement=float(displacement*(1+eps)+eps)
                cutoff=threshold*np.sqrt(1+self.metric_bound)+displacement
                radius=cutoff*np.sqrt(np.diag(self.baseline))
                lo=np.ceil(-delta-radius-eps*(abs(delta)+radius+1)).astype(int)
                hi=np.floor(-delta+radius+eps*(abs(delta)+radius+1)).astype(int)
                examined+=int(np.prod(np.maximum(hi-lo+1,0),dtype=object))
                if examined>max_images:raise ChemistryUnqualified('Bounded contact enumeration exceeded its image budget')
                for shift in product(*(range(int(a),int(b)+1) for a,b in zip(lo,hi))):
                    if i==j and shift<=(0,0,0):continue
                    d=delta+np.asarray(shift);length=float(np.linalg.norm(cartesian@d))
                    roundoff=eps*max(length,cutoff,1.)
                    lower=(length-displacement-roundoff)/np.sqrt(1+self.metric_bound)
                    if lower>=threshold:safe+=1;continue
                    upper=(length+displacement+roundoff)/np.sqrt(1-self.metric_bound)
                    if upper<threshold:impossible.append(dict(i=i,j=j,image=shift,upper_bound=upper,hard_min=threshold))
                    retained.append((d,directions,threshold,i,j,shift,lower,upper))
        self.delta=np.array([r[0] for r in retained]).reshape(-1,3)
        self.directions=np.array([r[1] for r in retained]).reshape(-1,3,self.dimension)
        self.threshold=np.array([r[2] for r in retained]);self.impossible=impossible
        self.certified_full_support=not retained;self.certified_zero_support=bool(impossible)
        self.receipt=dict(dimension=self.dimension,contacts_requiring_evaluation=len(retained),certified_safe_enumerated_contacts=safe,
            enumerated_images=examined,relative_reciprocal_metric_bound=self.metric_bound,
            reference_metric_condition_number=conditioning,arithmetic_margin=arithmetic,
            certified_full_support=self.certified_full_support,certified_zero_support=self.certified_zero_support,
            zero_support_witnesses=impossible,domain='Product of centered Euclidean balls in the declared coordinates',
            method='Triangle inequality plus a Frobenius upper bound on relative reciprocal-metric spectral norm',
            chemistry_catalog_fingerprint=catalog.fingerprint,not_a_scattering_approximation=True)
        self.fingerprint=fingerprint(dict(**self.receipt,base=base.tolist(),mapping=mapping.tolist(),baseline_a=np.asarray(baseline_a).tolist(),
            cell_mapping=cell_mapping.tolist(),species=labels,occupancy=occupancy.tolist(),lift=lift.tolist(),dimensions=self.dimensions,radii=self.radii))
        for array in [self.baseline,self.cell_directions,self.delta,self.directions,self.threshold]:array.flags.writeable=False

    def certifies_native(self,parameters,standardized,*,geometry_key,catalog_key):
        """Check a point with prebounded contacts only for its exact native lift.

        Contacts proven safe throughout the domain need no re-evaluation;
        every retained contact is evaluated at this point's current metric.
        Outside the bounded domain, mismatched contexts or reconstruction
        roundoff use the ordinary full-contact check instead.
        """
        z=np.asarray(standardized,float);parameters=np.asarray(parameters,float)
        if (geometry_key!=self.geometry_identity or catalog_key!=self.catalog_fingerprint
                or z.shape!=(self.dimension,) or parameters.shape!=(self.lift.shape[0],) or not np.isfinite(z).all()
                or not np.array_equal(parameters,self.lift@z)):
            return False
        if not all(np.linalg.norm(z[a:b])<=r for a,b,r in zip(self.offsets[:-1],self.offsets[1:],self.radii)):
            return False
        return bool(np.isfinite(self.log_kernel(z[None,:])[0]))

    def log_kernel(self,rows):
        rows=np.asarray(rows,float)
        if rows.ndim!=2 or rows.shape[1]!=self.dimension or not np.isfinite(rows).all():raise ValueError('Finite bounded-geometry parameter rows required')
        inside=np.ones(len(rows),bool)
        for a,b,r in zip(self.offsets[:-1],self.offsets[1:],self.radii):
            inside &= np.sum(rows[:,a:b]**2,axis=1)<=r*r+64*np.finfo(float).eps*max(b-a,1)*r*r
        result=np.full(len(rows),-np.inf)
        if self.certified_zero_support:return result
        if self.certified_full_support:result[inside]=0.;return result
        # Bound transient memory even for a large particle population.
        selected=np.flatnonzero(inside)
        for begin in range(0,len(selected),256):
            index=selected[begin:begin+256];z=rows[index]
            reciprocal=self.baseline+np.einsum('nk,kij->nij',z,self.cell_directions)
            factor=np.linalg.cholesky(reciprocal)
            fractional=self.delta[None,:,:]+np.einsum('cik,nk->nci',self.directions,z)
            transformed=np.linalg.solve(factor,fractional.transpose(0,2,1))
            squared=np.sum(transformed**2,axis=1)
            allowed=np.all(squared>=self.threshold[None,:]**2,axis=1)
            result[index[allowed]]=0.
        return result


class JointBoundedChemicalGeometry:
    """Several physical configurations constrained in one joint parameter space.

    Reference and ordered geometries may share columns. Conditioning each
    marginal independently would change this joint prior. Its normalization
    cancels in Z(reference,target)/Z(reference) only when both use this exact
    same joint prior and geometry policy.
    """
    def __init__(self,components):
        self.components=tuple(components)
        if (not self.components or len({name for name,_ in self.components})!=len(self.components)
                or any(not isinstance(name,str) or not name or not isinstance(c,BoundedChemicalGeometry)
                       for name,c in self.components)):
            raise ValueError('Distinct named bounded geometry components required')
        first=self.components[0][1];self.dimension=first.dimension
        self.dimensions=first.dimensions;self.radii=first.radii
        if any(c.dimensions!=self.dimensions or c.radii!=self.radii for _,c in self.components):
            raise ValueError('All geometries must use the same joint coordinate domain')
        self.certified_full_support=all(c.certified_full_support for _,c in self.components)
        self.certified_zero_support=any(c.certified_zero_support for _,c in self.components)
        self.receipt=dict(dimension=self.dimension,components={name:c.receipt for name,c in self.components},
            certified_full_support=self.certified_full_support,certified_zero_support=self.certified_zero_support,
            conditioning='One normalized joint prior; shared reference coordinates are not conditioned independently')
        self.fingerprint=fingerprint(dict(components=[(name,c.fingerprint) for name,c in self.components],
            conditioning=self.receipt['conditioning']))

    def log_kernel(self,rows):
        return sum(c.log_kernel(rows) for _,c in self.components)


class ConditionedPrior:
    """Exact independent draws and MH kernel for base_prior × hard chemistry.

    The unknown acceptance mass cancels in MH. SMC initialized from these
    normalized conditional draws estimates evidence under the conditional
    prior. Importance sampling with an external proposal additionally needs
    that acceptance mass; this class deliberately exposes no log_density.
    """
    def __init__(self,base_prior,chemistry,*,max_attempts=1000000):
        self.base=base_prior;self.chemistry=chemistry;self.bounds=np.asarray(base_prior.bounds,float).copy()
        if self.bounds.shape!=(chemistry.dimension,2) or type(max_attempts)is not int or max_attempts<1:
            raise ValueError('Compatible base-prior domain and bounded rejection budget required')
        self.bounds.flags.writeable=False;self.max_attempts=max_attempts
        self.fingerprint=fingerprint(dict(base=base_prior.fingerprint,chemistry=chemistry.fingerprint,policy='normalized_within_model_conditioning'))
        self.last_sampling=None

    def log_kernel(self,rows):return self.base.log_density(rows)+self.chemistry.log_kernel(rows)

    def sample(self,rng,count):
        if type(count)is not int or count<1:raise ValueError('Positive independent conditional sample count required')
        if self.chemistry.certified_zero_support:raise ChemistryUnqualified('Declared base-prior domain has zero empirical chemistry support')
        accepted=[];found=attempts=0
        while found<count and attempts<self.max_attempts:
            batch=min(max(32,count-found),self.max_attempts-attempts);rows=self.base.sample(rng,batch)
            keep=np.isfinite(self.chemistry.log_kernel(rows));accepted.append(rows[keep]);found+=int(keep.sum());attempts+=batch
        self.last_sampling=dict(attempts=attempts,accepted=found,requested=count,
            independent_draws=True,prior_mass_is_not_model_probability=True)
        if found<count:raise ChemistryUnqualified('Conditional prior sampling budget exhausted; no model exclusion')
        return np.concatenate(accepted)[:count]


def estimate_prior_mass(base_prior,chemistry,*,samples=8192,seed=0,confidence=.999):
    """Exact certificates or independent Bernoulli draws with binomial bounds.

    A finite all-success sample is not relabelled as mass=1 with certainty.
    The interval describes Monte Carlo uncertainty, not chemical validity.
    """
    from scipy.stats import beta
    if type(samples)is not int or samples<1 or not 0<confidence<1:
        raise ValueError('Positive fixed sample budget and confidence required')
    identity=dict(base_prior_fingerprint=base_prior.fingerprint,chemistry_fingerprint=chemistry.fingerprint)
    if chemistry.certified_full_support or chemistry.certified_zero_support:
        mass=float(chemistry.certified_full_support)
        return dict(**identity,method='bounded_geometry_certificate',mass=mass,interval=[mass,mass],samples=0,
            certified=True,monte_carlo_standard_error=0.,model_prior_odds_unchanged=True)
    rng=np.random.default_rng(seed);accepted=0
    for begin in range(0,samples,1024):
        rows=base_prior.sample(rng,min(1024,samples-begin))
        accepted+=int(np.isfinite(chemistry.log_kernel(rows)).sum())
    alpha=1-confidence
    lower=0. if accepted==0 else float(beta.ppf(alpha/2,accepted,samples-accepted+1))
    upper=1. if accepted==samples else float(beta.ppf(1-alpha/2,accepted+1,samples-accepted))
    mass=accepted/samples
    return dict(**identity,method='independent_fixed_budget_binomial',mass=mass,interval=[lower,upper],samples=samples,
        accepted=accepted,seed=seed,confidence=confidence,certified=False,
        monte_carlo_standard_error=float(np.sqrt(mass*(1-mass)/samples)),model_prior_odds_unchanged=True,
        log_normalizer_interval_width=float(np.log(upper/lower)) if lower>0 else None,
        interval_scope='Monte Carlo uncertainty about prior acceptance mass, not validity of empirical chemistry')
