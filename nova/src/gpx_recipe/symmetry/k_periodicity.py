"""Check actual translation characters before scoring a declared k family.

Subgroup services can include higher-symmetry alternatives whose magnetic cell
cannot carry the requested k. A larger cell can also contain a seed with only a
simpler periodicity. Neither is evidence for the requested propagation vector.
"""
import numpy as np
from ..commensurate_k import _vector, _operations


def star_vectors(k, operations):
    rotations, translations = _operations(operations)
    vector = _vector(k)
    result = []
    def same(a, b):
        difference = tuple(x-y for x,y in zip(a,b))
        return all(v.denominator == 1 for v in difference) and all(
            sum(x*t for x,t in zip(difference,translation)).denominator == 1
            for translation in translations)
    for rotation in rotations:
        rotated = tuple(sum(int(rotation[i,j])*vector[j] for j in range(3)) for i in range(3))
        for sign in (1,-1):
            value = tuple(sign*x for x in rotated)
            if not any(same(value,previous) for previous in result):result.append(value)
    return np.array([[float(v) for v in row] for row in result])


def compatible_arms(k, parent_operations, transformation, child_operations, spins, *, star=None):
    """Keep only star arms allowed by every pure magnetic translation.

Rows of the Bilbao matrix are child basis vectors in the original parent
coordinates. A unitary translation needs phase +1; an anti-translation needs
phase -1. Centred conventional cells are handled without reducing k modulo 1.
"""
    matrix = np.asarray(transformation,dtype=float)
    if matrix.shape != (3,3) or not np.isfinite(matrix).all():raise ValueError('Invalid magnetic basis')
    if len(child_operations) != len(spins):raise ValueError('Incomplete magnetic translations')
    translations = [(row,1) for row in matrix]
    for (rotation,translation),spin in zip(child_operations,spins):
        if np.array_equal(np.asarray(rotation),np.eye(3)):
            if spin not in (-1,1):raise ValueError('Invalid magnetic translation character')
            translations.append((matrix.T@np.asarray(translation,dtype=float),spin))
    vectors=star_vectors(k,parent_operations) if star is None else star
    return [q.tolist() for q in vectors
            if all(abs(np.exp(2j*np.pi*np.dot(q,t))-spin)<1e-7 for t,spin in translations)]


def projection_fraction(positions, moments, arms, parent_translations, *, species=None):
    """Fraction of the real-space moment field in the requested +/-k star.

Each chemical basis site gets independent complex Fourier coefficients. The
translation orbit includes parent centring vectors. No diffraction data enters
this check and no magnetic moment is altered to make the test pass.
"""
    positions=np.asarray(positions,dtype=float);moments=np.asarray(moments,dtype=float)
    arms=np.asarray(arms,dtype=float);translations=np.asarray(parent_translations,dtype=float)
    if positions.ndim!=2 or positions.shape[1]!=3 or moments.shape!=positions.shape or not np.isfinite(positions).all() or not np.isfinite(moments).all():raise ValueError('Invalid expanded magnetic structure')
    if arms.ndim!=2 or arms.shape[1]!=3 or not len(arms):return dict(fraction=0.,basis_sites=0,expanded_sites=len(positions))
    if translations.ndim!=2 or translations.shape[1]!=3 or not len(translations):raise ValueError('Parent translations are required')
    species=list(species) if species is not None else ['site']*len(positions)
    if len(species)!=len(positions):raise ValueError('Incomplete magnetic species inventory')
    groups={}
    for position,moment,kind in zip(positions,moments,species):
        choices=(position-translations)%1
        choices[np.isclose(choices,1,atol=1e-7,rtol=0)]=0
        basis=min(tuple(np.round(row,7)) for row in choices)
        groups.setdefault((kind,basis),[]).append((position-np.asarray(basis),moment))
    energy=0.;error=0.
    for values in groups.values():
        offsets=np.array([v[0] for v in values]);field=np.array([v[1] for v in values])
        design=np.exp(2j*np.pi*offsets@arms.T)
        coefficients=np.linalg.lstsq(design,field,rcond=1e-10)[0]
        energy+=float(np.sum(field*field));error+=float(np.sum(np.abs(field-design@coefficients)**2))
    if energy<1e-20:return dict(fraction=0.,basis_sites=len(groups),expanded_sites=len(positions))
    return dict(fraction=float(np.clip(1-error/energy,0.,1.)),basis_sites=len(groups),expanded_sites=len(positions))


def map_parent_orbits(positions, species, basis, translations, expected):
    """Recover translation identities despite native six-decimal rounding."""
    translations=np.asarray(translations);mapped=[];seen={};maximum_error=0.
    for point,kind in zip(positions,species):
        possibilities=[]
        for (reference_kind,token),reference in basis.items():
            if kind!=reference_kind:continue
            offsets=point-reference-translations
            errors=np.linalg.norm(offsets-np.rint(offsets),axis=1)
            index=int(np.argmin(errors));possibilities.append((float(errors[index]),token,reference,np.rint(offsets[index])+translations[index]))
        possibilities.sort(key=lambda item:item[0])
        if not possibilities or possibilities[0][0]>2e-5:raise ValueError('A generated magnetic atom cannot be mapped to the original nuclear translation orbit')
        if len(possibilities)>1 and possibilities[1][0]<2e-5:raise ValueError('Close nuclear basis sites make the magnetic translation mapping ambiguous')
        error,token,reference,offset=possibilities[0]
        maximum_error=max(maximum_error,error);mapped.append(reference+offset)
        translations_seen=seen.setdefault((kind,token),set());identity=tuple(np.round(offset,7))
        if identity in translations_seen:raise ValueError('A generated magnetic atom duplicates a parent translation')
        translations_seen.add(identity)
    if expected<1 or any(len(values)!=expected for values in seen.values()):
        raise ValueError('The generated magnetic structure does not cover every parent translation in its declared cell')
    return mapped,maximum_error


def audit_structure(parent, child, candidate, k):
    """Native-only expansion using the independently verified BNS operations."""
    from GSASII import GSASIIspc as S,GSASIIlattice as L
    parent_ops=S.AllOps(parent['General']['SGData'])[2]
    operations=[dict(rotation=r,translation=t) for r,t in parent_ops]
    child_ops=S.AllOps(child['General']['SGData'])[2];spins=child['General']['SGData']['SpnFlp']
    matrix=np.asarray(candidate['transformation_matrix'],dtype=float)
    arms=compatible_arms(k,operations,matrix,child_ops,spins)
    if not arms:return dict(passed=False,reason='The magnetic translation lattice cannot carry any arm of the requested k star',compatible_arms=[],fraction=0.)
    a=L.cell2AB(child['General']['Cell'][1:7])[0];inverse=np.linalg.inv(a)
    native_to_cartesian=a@np.diag(1/np.asarray(child['General']['Cell'][1:4]))
    cx,ct,_,_=child['General']['AtomPtrs'];expanded={}
    for atom in child['Atoms']:
        position=np.asarray(atom[cx:cx+3],dtype=float);moment=native_to_cartesian@atom[cx+4:cx+7]
        # Follow GSAS-II's unique-site convention, including its documented
        # trigonal/hexagonal special-position handling. Expanding rounded 1/3
        # coordinates with exact arithmetic invents three almost-coincident
        # atoms where the native project declares one special-position atom.
        for point,code,_,spin in S.GenAtom(position,child['General']['SGData']):
            rotation=np.asarray(child['General']['SGData']['SGOps'][abs(int(code))%100-1][0])
            if code<0:rotation=-rotation
            point=np.asarray(point)%1
            vector=spin*np.linalg.det(rotation)*a@rotation@inverse@moment
            key=(str(atom[ct]),tuple(np.round(point,7)))
            previous=next((v for old,v in expanded.items() if old[0]==key[0] and np.max(np.abs((v[0]-point+.5)%1-.5))<2e-4),None)
            if previous is not None:
                if not np.allclose(previous[1],vector,atol=1e-7,rtol=0):raise ValueError('Magnetic symmetry mates disagree at the same chemical site')
                continue
            expanded[key]=(point,vector)
            if len(expanded)>10000:raise ValueError('Expanded magnetic structure exceeds the periodicity audit budget')
    translations=np.asarray([t for r,t in parent_ops if np.array_equal(r,np.eye(3))])
    # TransformPhase rounds child positions to six decimals. Recover each
    # translation from the *original nuclear basis*, rather than inventing
    # separate basis sites from those rounding errors. The GPX is never moved.
    import re
    element=lambda value:re.match(r'^[A-Z][a-z]?',str(value)).group()
    pcx,pct,_,_=parent['General']['AtomPtrs'];basis={}
    for atom in parent['Atoms']:
        for point,_,_,_ in S.GenAtom(atom[pcx:pcx+3],parent['General']['SGData']):
            point=np.asarray(point)%1
            choices=(point-translations)%1;choices[np.isclose(choices,1,atol=1e-7,rtol=0)]=0
            canonical=min(tuple(np.round(row,7)) for row in choices)
            basis[(element(atom[pct]),canonical)]=np.asarray(canonical)
    points=[matrix.T@row[0]-np.asarray(candidate['origin_shift']) for row in expanded.values()]
    number=abs(np.linalg.det(matrix))*len(translations);expected=int(round(number))
    if not np.isclose(number,expected,atol=1e-7,rtol=0):raise ValueError('The child cell is not commensurate with the parent translation lattice')
    points,maximum_error=map_parent_orbits(points,[element(key[0]) for key in expanded],basis,translations,expected)
    kinds=[key[0] for key in expanded]
    report=projection_fraction(points,[row[1] for row in expanded.values()],arms,translations,species=kinds)
    report.update(maximum_parent_mapping_error=maximum_error,translations_per_basis_site=expected,
                  site_generation='Native GSAS-II unique-site and special-position conventions; input GPX unchanged')
    report.update(compatible_arms=arms,passed=report['fraction']>=1-1e-7)
    if not report['passed']:report['reason']='The starting moments contain a different translation periodicity; this profile cannot be scored as the requested k'
    return report
