"""Evaluate Cluster B (same layout as Cluster A — reuse evaluate_config)."""

import argparse
import os
import warnings

from multiomics.code.cluster_a_contrastive_dec.evaluate import evaluate_config

warnings.filterwarnings('ignore')


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_cluster_b'.format(disease)
    if args.config:
        folder = args.config
    else:
        import json
        best = os.path.join(parent, 'best_config.json')
        if not os.path.exists(best):
            raise FileNotFoundError(best)
        folder = json.load(open(best))['config']
    path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(disease, path)
    for k, v in metrics.items():
        print('  {}: {}'.format(k, v))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--disease', default='brca')
    p.add_argument('--config', default='')
    args, unknown = p.parse_known_args()
    if unknown:
        raise ValueError(unknown)
    main(args)
