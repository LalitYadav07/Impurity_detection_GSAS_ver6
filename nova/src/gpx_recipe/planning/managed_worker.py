"""Worker-only dispatch into the existing, shared native planner engines."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from gpx_recipe.execution import write_json
from .managed import NATIVE, RESULT_SCHEMA, read_plan


def execute(operation, payload):
    request=read_plan(payload['plan'],operation)
    output=Path(payload['destination'])
    if output.exists() and any(output.iterdir()):raise ValueError('Planning outputs are immutable')
    output.mkdir(parents=True,exist_ok=True)
    if operation in NATIVE:
        from .magnetic import execute as native
        result=native(NATIVE[operation],request,output)
        write_json(output/'result.json',result)
    else:
        from .worker import run
        result=run(request,output)
    return dict(schema=RESULT_SCHEMA,operation=operation,output=str(output),result=result)


def main():
    operation,request_path,response=sys.argv[1:]
    result=execute(operation,json.loads(Path(request_path).read_text()))
    write_json(Path(response),result)


if __name__=='__main__':main()
