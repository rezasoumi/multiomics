"""Evaluate Method 1 + VICReg ablation models."""

import os
import argparse
import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from multiomics.code.loading_data import load_data
from multiomics.code import util
import multiomics.code.evaluation as evaluation
from multiomics.code.method1_vicreg_ablation.model import SharedAndSpecificEmbedding

warnings.filterwarnings('ignore')

NUM_CLUST = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _truth_from_label(disease, label):
    if disease == 'coad':
        return label.flatten().astype('int')
    if disease == 'lihc':
        lst = label[:, 0].flatten()
        unique_vals = list(set(lst))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return np.asarray([mapping[val] for val in lst])
    if disease == 'kirc':
        return label[:, 1].flatten().astype('int')
    return label.flatten()


def evaluate_config(disease, config_dir, plot=False):
    num_clust = NUM_CLUST[disease]
    view1_data, view2_data, view3_data, _, _ = load_data(disease)

    data = np.load(os.path.join(config_dir, 'test_data_{}.npy'.format(disease)))
    label = np.load(
        os.path.join(config_dir, 'test_label_{}.npy'.format(disease)),
        allow_pickle=True,
    )

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    model.load_state_dict(
        torch.load(os.path.join(config_dir, 'model_{}'.format(disease)), map_location='cpu')
    )
    model.eval()
    setup_seed(2)

    view1 = torch.tensor(data[:, :view1_data.shape[1]], dtype=torch.float32)
    view2 = torch.tensor(
        data[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]],
        dtype=torch.float32,
    )
    view3 = torch.tensor(
        data[:, view1_data.shape[1] + view2_data.shape[1]:],
        dtype=torch.float32,
    )

    with torch.no_grad():
        out = model(view1, view2, view3)
    shared_common = (out['shared'][1] + out['shared'][2] + out['shared'][3]) / 3
    final_embedding = torch.cat(
        (out['specific'][1], out['specific'][2], out['specific'][3], shared_common),
        dim=1,
    ).numpy()

    truth = _truth_from_label(disease, label)
    if plot:
        util.plot_with_path(
            final_embedding,
            truth,
            os.path.join(config_dir, 'final_em'),
            'Method1VicRegAblation',
        )

    km = KMeans(n_clusters=num_clust, random_state=42)
    y_pred = km.fit_predict(final_embedding)
    nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)

    X_tr, X_te, y_tr, y_te = train_test_split(
        final_embedding, truth, test_size=0.25, random_state=12
    )
    knn = KNeighborsClassifier(n_neighbors=num_clust)
    knn.fit(X_tr, y_tr)
    knn_acc = float(accuracy_score(y_te, knn.predict(X_te)))

    return {
        'nmi': float(nmi_),
        'ari': float(ari_),
        'f_score': float(f_score_),
        'acc': float(acc_),
        'v_measure': float(v_),
        'ch_index': float(ch),
        'knn_acc': knn_acc,
    }


def main(args):
    disease = args.disease
    parent = '../../results/models_{}_method1_vicreg'.format(disease)

    if args.config:
        folder = args.config
    else:
        ls = [{'loss': 1e12, 'config': None}]
        if os.path.isdir(parent):
            for config in os.listdir(parent):
                f = os.path.join(parent, config, 'loss.npy')
                if os.path.exists(f):
                    ls.append({'loss': float(np.load(f)), 'config': config})
        best = min(ls, key=lambda x: x['loss'])
        if best['config'] is None:
            raise FileNotFoundError('No trained models under {}'.format(parent))
        folder = best['config']

    desired_path = os.path.join(parent, folder)
    print('Using config:', folder)
    metrics = evaluate_config(disease, desired_path, plot=True)
    print(
        '\n' + ' ' * 8
        + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
          'ch_index: %.4f  <==|'
        % (
            metrics['nmi'],
            metrics['ari'],
            metrics['f_score'],
            metrics['acc'],
            metrics['v_measure'],
            metrics['ch_index'],
        )
    )
    print('kNN acc: {:.2f}'.format(metrics['knn_acc']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Method 1 VICReg ablation')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
