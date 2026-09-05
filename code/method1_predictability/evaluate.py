"""Evaluate a trained Method 1 model (clustering + kNN), MOCSS-style."""

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
from multiomics.code.method1_predictability.model import SharedAndSpecificEmbedding

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main(args):
    method = 'Method1Predictability'
    disease = args.disease
    num_clust = {'lihc': 2, 'coad': 4, 'kirc': 2, 'brca': 5}[disease]

    view1_data, view2_data, view3_data, _, _ = load_data(disease)

    parent = '../../results/models_{}_method1'.format(disease)
    ls = [{'loss': 1e12, 'config': None}]
    if os.path.isdir(parent):
        for config in os.listdir(parent):
            f = os.path.join(parent, config, 'loss.npy')
            if os.path.exists(f):
                ls.append({'loss': float(np.load(f)), 'config': config})
    best = min(ls, key=lambda x: x['loss'])
    if best['config'] is None:
        raise FileNotFoundError('No trained Method 1 models under {}'.format(parent))

    folder = args.config if args.config else best['config']
    desired_path = os.path.join(parent, folder)
    print('Using config:', folder)

    data = np.load(desired_path + '/test_data_{}.npy'.format(disease))
    label = np.load(desired_path + '/test_label_{}.npy'.format(disease), allow_pickle=True)

    model = SharedAndSpecificEmbedding(
        view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
        n_units_1=[512, 256, 128, 32],
        n_units_2=[512, 256, 128, 32],
        n_units_3=[256, 128, 64, 32],
        mlp_size=[32, 8],
    )
    model.load_state_dict(
        torch.load(desired_path + '/model_{}'.format(disease), map_location='cpu')
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

    if disease == 'coad':
        truth = label.flatten().astype('int')
    elif disease == 'lihc':
        lst = label[:, 0].flatten()
        unique_vals = list(set(lst))
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        truth = np.asarray([mapping[val] for val in lst])
    elif disease == 'kirc':
        truth = label[:, 1].flatten().astype('int')
    else:
        truth = label.flatten()

    util.plot_with_path(final_embedding, truth, desired_path + '/final_em', method)
    km = KMeans(n_clusters=num_clust, random_state=42)
    y_pred = km.fit_predict(final_embedding)
    nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
    print(
        '\n' + ' ' * 8
        + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
          'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch)
    )

    X_train, X_test, y_train, y_test = train_test_split(
        final_embedding, truth, test_size=0.25, random_state=12
    )
    knn = KNeighborsClassifier(n_neighbors=num_clust)
    knn.fit(X_train, y_train)
    print('kNN acc: {:.2f}'.format(accuracy_score(y_test, knn.predict(X_test))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Method 1')
    parser.add_argument('--disease', type=str, default='brca')
    parser.add_argument('--config', type=str, default='', help='Subfolder under models_*_method1')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    main(args)
