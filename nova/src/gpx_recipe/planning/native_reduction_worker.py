"""One native Mantid operation per Linux process; never import archive scripts."""
import argparse
import json
from pathlib import Path

from .native_reduction import execute_native_operation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--read-only-inputs', action='store_true',
                        help='Use one exact flat input bundle on a kernel-enforced read-only mount')
    args = parser.parse_args()
    if not 0 < args.request.stat().st_size <= 2 * 1024**2:
        raise ValueError('A bounded native operation request is required')
    result = execute_native_operation(json.loads(args.request.read_text()), args.output,
                                      input_mode='read_only_mount' if args.read_only_inputs else 'copy')
    print(json.dumps({'operation_identity': result['operation_identity'],
                      'saved_products': len(result['products']),
                      'experimental_accuracy_certified': False}))


if __name__ == '__main__':
    main()
