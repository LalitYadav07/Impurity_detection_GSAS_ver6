"""Conditional transfer from a physical V law to the planner's elastic law.

Both supplied predictions must use the same source, efficiency, reference
geometry and observation operator. Their ratio changes the reference scattering
law while retaining the measured ON/OFF response and its covariance. It is a
local multiplicative discrepancy approximation, not an inelastic unfolding of
each observed neutron. Reference-law/source/geometry uncertainty is separate.
"""
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from .focused_reduction import overlap_map


@dataclass(frozen=True)
class ReferenceLawTransfer:
    profile_id: str
    bundle_sha256: str
    d_edges_a: np.ndarray
    elastic_reference_rate: np.ndarray
    physical_reference_rate: np.ndarray
    provenance: dict

    def __post_init__(self):
        edges = np.array(self.d_edges_a, dtype=float, copy=True)
        elastic = np.array(self.elastic_reference_rate, dtype=float, copy=True)
        physical = np.array(self.physical_reference_rate, dtype=float, copy=True)
        if (edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all()
                or np.any(np.diff(edges) <= 0) or edges[0] <= 0
                or elastic.ndim != 2 or elastic.shape != physical.shape
                or elastic.shape[1] != len(edges)-1
                or any(not np.isfinite(a).all() or np.any(a < 0) for a in (elastic, physical))):
            raise ValueError('Reference-law predictions need matching finite nonnegative angular bin rates')
        if not self.profile_id or not self.bundle_sha256 or not self.provenance:
            raise ValueError('Reference-law transfer requires configuration and input provenance')
        for key, value in [('d_edges_a', edges), ('elastic_reference_rate', elastic),
                           ('physical_reference_rate', physical)]:
            value.setflags(write=False)
            object.__setattr__(self, key, value)

    def factors(self, profile_id, bundle_sha256, target_edges_a, required_cells):
        """Ratio of projected model bin integrals, never an average of ratios.

        Missing physical support in a required measured cell is an error. Empty
        unused cells get an identity factor; they never acquire invented counts.
        """
        return self.rate_ratio(self.elastic_reference_rate, profile_id, bundle_sha256,
            target_edges_a, required_cells, unused_default=1.)

    def rate_ratio(self, numerator, profile_id, bundle_sha256, target_edges_a,
                   required_cells, *, unused_default=0.):
        """Project a positive source numerator against this physical reference."""
        numerator = np.asarray(numerator, float)
        if (numerator.shape != self.physical_reference_rate.shape or not np.isfinite(numerator).all()
                or np.any(numerator < 0) or not np.isfinite(unused_default) or unused_default < 0):
            raise ValueError('Source numerator must match the finite nonnegative physical reference')
        if profile_id != self.profile_id or bundle_sha256 != self.bundle_sha256:
            raise ValueError('Reference-law transfer belongs to another calibration bundle')
        target = np.asarray(target_edges_a, float)
        required = np.asarray(required_cells, bool)
        if (target.ndim != 1 or len(target) < 2 or not np.isfinite(target).all()
                or np.any(np.diff(target) <= 0)
                or required.shape != (self.elastic_reference_rate.shape[0], len(target)-1)):
            raise ValueError('Reference-law target grid and required angular cells must match')
        # Zero extension is used only to construct the overlap operator. It does
        # not establish support: every required target cell must be fully inside.
        edges = self.d_edges_a
        left, right = int(target[0] < edges[0]), int(target[-1] > edges[-1])
        padded = np.r_[target[:left], edges, target[-right:] if right else []]
        projection = overlap_map(padded, target)
        # Sparse projection acts on bins, with groups supplied as columns.
        projected = []
        for rates in (numerator, self.physical_reference_rate):
            padded_rates = np.pad(rates, ((0, 0), (left, right)))
            projected.append(np.asarray(projection @ padded_rates.T).T)
        elastic, physical = projected
        covered = (target[:-1] >= edges[0]) & (target[1:] <= edges[-1])
        if np.any(required & (~covered[None, :] | (physical <= 0))):
            raise ValueError('Physical reference law does not cover every required measured cell')
        result = np.full_like(physical, unused_default)
        valid = covered[None, :] & (physical > 0)
        result[valid] = elastic[valid]/physical[valid]
        if not np.isfinite(result).all():
            raise ValueError('Reference-law ratio exceeds its finite numerical domain')
        return result


def sample_response_fingerprint(metadata, sample):
    """Exact compiled-material/geometry identity, independent of local paths."""
    material = {key: metadata[key] for key in ('cell_mass_da', 'cell_volume_a3',
        'composition', 'neutron_table', 'mixture', 'phase_mass_fractions', 'nuclear_cell_multiplier')}
    material['sources'] = {key: row['sha256'] for key, row in metadata['sources'].items()}
    material['sample'] = sample
    return hashlib.sha256(json.dumps(material, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SampleElasticTransfer:
    """Matched geometry for elastic scattering, using an isotropic unit law.

    isotropic_rate is the modeled response of the WHOLE specimen to a fictitious
    isotropic elastic cross section normalizer_b per declared crystallographic
    cell. Its real removal/illumination/path weights, source and efficiency must
    match the physical reference calculation. The crystallographic profile is
    divided by this normalizer before multiplication; specimen amount and
    attenuation must not be applied again. This is still a bin-averaged transfer,
    not an exact angular/wavelength convolution of every Bragg reflection.
    """
    reference_law: ReferenceLawTransfer
    isotropic_rate: np.ndarray
    normalizer_b: float
    sample_fingerprint: str
    provenance: dict

    def __post_init__(self):
        rate = np.array(self.isotropic_rate, dtype=float, copy=True)
        if (not isinstance(self.reference_law, ReferenceLawTransfer)
                or rate.shape != self.reference_law.physical_reference_rate.shape
                or not np.isfinite(rate).all() or np.any(rate < 0)
                or isinstance(self.normalizer_b, (bool, np.bool_, str))
                or not np.isscalar(self.normalizer_b) or not np.isfinite(self.normalizer_b)
                or not 0 < self.normalizer_b <= 1e6
                or not isinstance(self.sample_fingerprint, str) or len(self.sample_fingerprint) != 64
                or not self.provenance
                or self.provenance.get('isotropic_elastic_unit_law') is not True
                or self.provenance.get('whole_specimen_crystallographic_cell_basis') is not True
                or self.provenance.get('coherent_bragg_included') is not False):
            raise ValueError('Elastic transfer requires a matched whole-specimen isotropic unit law and cell basis')
        rate.setflags(write=False)
        object.__setattr__(self, 'isotropic_rate', rate)
        object.__setattr__(self, 'normalizer_b', float(self.normalizer_b))

    def factors(self, profile_id, bundle_sha256, target_edges_a, required_cells,
                metadata, sample):
        if sample_response_fingerprint(metadata, sample) != self.sample_fingerprint:
            raise ValueError('Elastic transfer does not match the requested structure, isotope, mixture or geometry')
        return self.reference_law.rate_ratio(self.isotropic_rate, profile_id, bundle_sha256,
            target_edges_a, required_cells)


@dataclass(frozen=True)
class SampleDiffuseTransfer:
    """A matched material's diffuse/inelastic source, excluding coherent Bragg.

    This replaces the thermal incoherent source, including its counting noise.
    Its uncertain measured reference is shared with the Bragg source. A changed
    structure, isotope, mixture or specimen requires a new compiled response.
    """
    reference_law: ReferenceLawTransfer
    diffuse_rate: np.ndarray
    sample_fingerprint: str
    provenance: dict

    def __post_init__(self):
        rate = np.array(self.diffuse_rate, dtype=float, copy=True)
        if (rate.shape != self.reference_law.physical_reference_rate.shape
                or not np.isfinite(rate).all() or np.any(rate < 0)
                or not isinstance(self.sample_fingerprint, str) or len(self.sample_fingerprint) != 64
                or not self.provenance
                or self.provenance.get('coherent_bragg_included') is not False
                or self.provenance.get('includes_incoherent_elastic') is not True):
            raise ValueError('Diffuse response requires a matched sample and an explicit non-Bragg component contract')
        rate.setflags(write=False)
        object.__setattr__(self, 'diffuse_rate', rate)

    def factors(self, profile_id, bundle_sha256, target_edges_a, required_cells,
                metadata, sample):
        if sample_response_fingerprint(metadata, sample) != self.sample_fingerprint:
            raise ValueError('Diffuse response does not match the requested structure, isotope, mixture or geometry')
        return self.reference_law.rate_ratio(self.diffuse_rate, profile_id, bundle_sha256,
            target_edges_a, required_cells)
