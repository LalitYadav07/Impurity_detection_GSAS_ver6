"""Portable origin *references*, never scientific authority by themselves."""
import re
from urllib.parse import urlsplit,urlunsplit

from .certificates import checksum


def site(value):
    if not isinstance(value,str) or not 1<=len(value)<=512:
        raise ValueError('Bounded Galaxy site identity required')
    parsed=urlsplit(value)
    if (parsed.scheme not in {'http','https'} or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path.endswith('//')):
        raise ValueError('Galaxy origin must not contain credentials, query or fragment')
    return urlunsplit((parsed.scheme,parsed.netloc,parsed.path.rstrip('/'),'',''))


def validate_references(value):
    if not isinstance(value,list) or len(value)>5000:raise ValueError('Bounded native origin references required')
    seen=set()
    for row in value:
        if (not isinstance(row,dict) or set(row)!={'site','dataset_id','sha256','entries'}
                or site(row['site'])!=row['site'] or not isinstance(row['dataset_id'],str)
                or not re.fullmatch('[A-Za-z0-9_-]{1,128}',row['dataset_id']) or not checksum(row['sha256'])
                or not isinstance(row['entries'],list) or not 1<=len(row['entries'])<=256
                or any(not checksum(v) for v in row['entries']) or len(set(row['entries']))!=len(row['entries'])
                or (row['site'],row['dataset_id']) in seen):
            raise ValueError('Invalid or duplicate native origin reference')
        seen.add((row['site'],row['dataset_id']))
    return value


def select(references,entries):
    validate_references(references);wanted=set(entries)
    return [dict(row,entries=sorted(wanted.intersection(row['entries']))) for row in references
            if wanted.intersection(row['entries'])]
