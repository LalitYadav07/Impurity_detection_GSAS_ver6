"""Bounded, credential-free preparation packet used by native workers."""
import json
from pathlib import Path, PurePosixPath
import stat
import zipfile

MAX_PACKET_BYTES = 512 * 1024 * 1024
MAX_MEMBERS = 128


def extract_preparation_packet(packet, destination):
    """Extract only regular flat inputs; never restore caller-chosen paths."""
    packet, destination = Path(packet), Path(destination)
    if packet.stat().st_size > MAX_PACKET_BYTES:
        raise ValueError('Preparation packet exceeds 512 MiB')
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(packet) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= MAX_MEMBERS or sum(m.file_size for m in members) > MAX_PACKET_BYTES:
            raise ValueError('Preparation packet has too many or oversized inputs')
        names = set()
        for member in members:
            name = member.filename
            mode = member.external_attr >> 16
            if (not name or len(PurePosixPath(name).parts) != 1 or name in {'.', '..'}
                    or '/' in name or '\\' in name or member.is_dir()
                    or name in names or member.flag_bits & 1
                    or (stat.S_IFMT(mode) not in {0, stat.S_IFREG})):
                raise ValueError('Preparation packet contains an unsafe or duplicate input')
            names.add(name)
        if 'preparation.json' not in names:
            raise ValueError('Preparation packet has no manifest')
        if archive.getinfo('preparation.json').file_size > 2 * 1024 * 1024:
            raise ValueError('Preparation manifest exceeds 2 MiB')
        for member in members:
            with archive.open(member) as source, (destination / member.filename).open('xb') as target:
                remaining = member.file_size
                while remaining:
                    data = source.read(min(1024 * 1024, remaining))
                    if not data:
                        raise ValueError('Preparation packet is truncated')
                    target.write(data); remaining -= len(data)
                if source.read(1):
                    raise ValueError('Preparation packet exceeds its declared size')
    return json.loads((destination / 'preparation.json').read_text())


def prepare_packet(packet, directory):
    from .preparation import prepare_model
    directory = Path(directory)
    inputs = directory / 'inputs'
    spec = extract_preparation_packet(packet, inputs)
    return prepare_model(spec, inputs, directory / 'model')
