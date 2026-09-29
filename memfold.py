#!/usr/bin/env python3
"""MemFold: prepare data, extract memory, train, and evaluate."""
import argparse
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'prepare': {
        'data': 'scripts/prepare_data.py',
        'split': 'scripts/split_data.py',
        'writer-inputs': 'scripts/prepare_writer_inputs.py',
        'writer-targets': 'scripts/prepare_writer_targets.py',
        'memory-data': 'scripts/prepare_memory_data.py',
        'generated-data': 'scripts/prepare_generated_data.py',
        'reasoning': 'scripts/prepare_reasoning.py',
    },
    'extract': 'scripts/extract_memory.py',
    'generate': 'scripts/generate_memory.py',
    'encode': 'scripts/encode_memory.py',
    'train': {
        'writer': 'scripts/train_writer.py',
        'compressor': 'scripts/train_compressor.py',
        'warmup': 'scripts/warmup_compressor.py',
        'reasoning': 'scripts/train_reasoning.py',
        'reader': 'scripts/train_reader.py',
        'optimize': 'scripts/optimize.py',
    },
    'evaluate': 'scripts/evaluate.py',
    'tokens': 'scripts/count_tokens.py',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name, target in COMMANDS.items():
        child = commands.add_parser(name, add_help=isinstance(target, dict))
        if isinstance(target, dict):
            actions = child.add_subparsers(dest='action', required=True)
            for action, script in target.items():
                actions.add_parser(action, add_help=False).set_defaults(script=script)
        else:
            child.set_defaults(script=target)
    args, remaining = parser.parse_known_args()
    script = ROOT / args.script
    sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts'), str(ROOT)]
    sys.argv = [str(script), *remaining]
    runpy.run_path(str(script), run_name='__main__')


if __name__ == '__main__':
    main()
