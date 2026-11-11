"""Render reviewed preparation-capable Galaxy tools from immutable images.

Rendering writes local review files. It never installs tools or deploys Galaxy.
"""
import argparse
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def render(image, output_directory):
    if not re.fullmatch(r'[a-z0-9][a-z0-9.:-]*/[a-z0-9._/-]+(?::models-[a-f0-9]{40}|@sha256:[a-f0-9]{64})', image):
        raise ValueError('Use the successful Model image full-commit tag or registry digest')
    templates = Path(__file__).resolve().parents[1]/'galaxy'
    provenance = (templates/'radar_pd_preparation_provenance.xml.in').read_text().rstrip()
    rendered = {}
    for name in ('model_campaign', 'live_snapshot'):
        text = (templates/f'radar_pd_{name}.xml.in').read_text()
        text = text.replace('@IMMUTABLE_MODELS_IMAGE@', image).replace('@PREPARATION_PROVENANCE@', provenance)
        if '@IMMUTABLE_' in text or '@PREPARATION_' in text:
            raise ValueError('An unresolved deployment placeholder remains')
        ET.fromstring(text)
        rendered[name] = text
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in rendered.items():
        path = output/f'radar_pd_{name}.xml'
        if path.is_symlink():
            raise ValueError('Refusing a symlinked Galaxy definition')
        path.write_text(text)
        paths.append(path)
    return paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image')
    parser.add_argument('output_directory')
    args = parser.parse_args()
    for path in render(args.image, args.output_directory):
        print(path)
