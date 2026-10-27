"""Carry bounded scientific input notices into durable Galaxy results.

An input marker may add review notices, never certify an input or clear a
pipeline warning. Only the explicit small GSAS header is inspected.
"""
import json
from pathlib import Path

PREFIX=b'# RADAR-PD input provenance: '


def read_input_provenance(path):
    with Path(path).open('rb') as stream:
        for _ in range(64):
            line=stream.readline(8193)
            if not line or line.startswith(b'BANK'):break
            if len(line)>8192:
                if line.startswith(PREFIX):raise ValueError('Scientific input provenance exceeds its size limit')
                return {}  # Unmarked input formats keep their existing reader contract.
            if not line.startswith(PREFIX):continue
            payload=json.loads(line[len(PREFIX):])
            if not isinstance(payload,dict) or payload.get('schema')!='radar-pd-input-provenance/v1':
                raise ValueError('Unsupported scientific input provenance')
            warnings=payload.get('warnings')
            if not isinstance(warnings,list) or len(warnings)>16 or any(not isinstance(w,dict) or not isinstance(w.get('message'),str) or len(w['message'])>1000 for w in warnings):
                raise ValueError('Invalid scientific input notices')
            if 'acquisition' in payload:
                try:
                    from .ndip_acquisition import validate_acquisition
                except ImportError:
                    from ndip_acquisition import validate_acquisition
                payload['acquisition'] = validate_acquisition(payload['acquisition'])
            return payload
    return {}
