"""Explicitly reviewed mapping for a live source with no embedded calibration."""
from __future__ import annotations
import math,re


def reviewed_mapping(*,source,receipt,bank,export_difc,normalization,producer_sha256,evidence,accepted):
    if accepted is not True:raise ValueError('Confirm that the producer mapping was checked')
    if isinstance(bank,bool) or not str(bank).isdigit() or int(bank)<1:raise ValueError('A physical detector bank is required')
    expected=str(source.get('bank') or '')
    if expected and str(int(bank))!=expected:raise ValueError('Live bank differs from the selected source GPX bank')
    if isinstance(export_difc,bool):raise ValueError('Enter the producer export conversion factor')
    try:factor=float(export_difc)
    except (ValueError,TypeError):raise ValueError('Enter the producer export conversion factor') from None
    if not math.isfinite(factor) or factor<=0:raise ValueError('Export conversion factor must be finite and positive')
    if not re.fullmatch('[a-f0-9]{64}',str(producer_sha256)):raise ValueError('Provide the checked producer configuration SHA-256')
    if len(str(normalization).strip())<5:raise ValueError('Describe the producer normalization convention')
    if len(str(evidence).strip())<12:raise ValueError('Record where the bank/export mapping was verified')
    return dict(schema='radar-live-user-mapping/v1',source_id=source['id'],source_gpx_sha256=source.get('source_sha256'),
                snapshot_sha256=receipt['content_sha256'],identity=receipt['identity'],physical_bank=int(bank),
                export_geometry=dict(kind='linear_tof_from_d',microseconds_per_angstrom=factor),
                normalization=str(normalization).strip(),producer_config_sha256=producer_sha256,
                evidence=str(evidence).strip(),mapping_kind='user_reviewed',scientifically_qualified=False,
                native_preparation_required=True,forecast_enabled=False)
