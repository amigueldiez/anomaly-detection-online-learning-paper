#!/usr/bin/env python3
"""Merge the per-model CIC multi-seed summaries into one JSON file."""

import json
from pathlib import Path

OUTPUT_DIR = Path('resultados')
MODELS = ('SAD', 'LODA', 'OCSVM')
MERGED_PATH = OUTPUT_DIR / 'multisemilla_cic_ids2018.json'


def main():
    merged = None
    for model in MODELS:
        path = OUTPUT_DIR / f'multisemilla_cic_ids2018_{model}.json'
        if not path.exists():
            raise FileNotFoundError(f'Missing summary: {path}')
        data = json.loads(path.read_text())

        if merged is None:
            merged = {
                'protocol': {
                    'num_executions': data['protocol']['num_executions'],
                    'seeds': data['protocol']['seeds'],
                    'split': data['protocol']['split'],
                    'datasets': data['protocol']['datasets'],
                    'models': {},
                },
                'results': {},
                'total_seconds': 0.0,
            }

        merged['protocol']['models'].update(data['protocol']['models'])
        merged['results'].update(data['results'])
        merged['total_seconds'] = max(
            merged['total_seconds'], data.get('total_seconds', 0.0),
        )

    MERGED_PATH.write_text(json.dumps(merged, indent=2) + '\n')
    print(f'Merged summary saved to {MERGED_PATH}')


if __name__ == '__main__':
    main()
