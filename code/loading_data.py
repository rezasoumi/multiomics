import os

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split


def load_or_rebuild_test_split(disease, config_dir, random_state=1):
    """Load saved test split, or rebuild the same 80/20 split used in training."""
    data_path = os.path.join(config_dir, 'test_data_{}.npy'.format(disease))
    label_path = os.path.join(config_dir, 'test_label_{}.npy'.format(disease))
    if os.path.exists(data_path) and os.path.exists(label_path):
        return (
            np.load(data_path),
            np.load(label_path, allow_pickle=True),
        )
    _, _, _, view_train_concatenate, y_true = load_data(disease)
    _, X_test, _, y_test = train_test_split(
        view_train_concatenate, y_true, test_size=0.2, random_state=random_state
    )
    return X_test, y_test


def load_data(disease):
    if disease == 'tcga_brca':
        view1_data = pd.read_csv('../../data/tcga_brca/mrna_txt_transpose/mrna_counts_most_variant.csv', header=None)  # mRNA
        view2_data = pd.read_csv('../../data/tcga_brca/methylation_txt_transpose/methyl_most_variant.csv', header=None)  # DNAmeth
        view3_data = pd.read_csv('../../data/tcga_brca/mirna_txt_transpose/mirna_most_variant.csv', header=None)  # miRNA
        label = pd.read_csv('../../code/tcga_brca_multiomics_mapping.csv', header=None)
    elif disease == 'brca':
        view1_data = pd.read_csv('../../data/brca/1_all.csv', header=None)  # mRNA
        view2_data = pd.read_csv('../../data/brca/2_all.csv', header=None)  # DNAmeth
        view3_data = pd.read_csv('../../data/brca/3_all.csv', header=None)  # miRNA
        label = pd.read_csv('../../data/brca/labels_all.csv', header=None)
    elif disease == 'lihc':
        view1_data = pd.read_csv('../../data/LIHC/data/gene.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/LIHC/data/methyl.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/LIHC/data/miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/LIHC/label.csv', header=None)
        label = label.to_numpy()[1:, 1:]
    elif disease == 'coad':
        view1_data = pd.read_csv('../../data/COAD/COAD_mRNA.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/COAD/COAD_Methy.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/COAD/COAD_miRNA.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/COAD/COAD_label.csv', header=None)
        label = label.to_numpy()[1:, 0]
        label[label == 'CIN'] = 0
        label[label == 'GS'] = 1
        label[label == 'MSI'] = 2
        label[label == 'POLE'] = 3
    elif disease == 'kirc':
        view1_data = pd.read_csv('../../data/KIRC/data/gene1.csv')
        view1_data = view1_data.T # mRNA
        view1_data = view1_data.iloc[1:, 1:]
        view2_data = pd.read_csv('../../data/KIRC/data/methyl.csv', header=None)
        view2_data = view2_data.T # DNAmeth
        view2_data = view2_data.iloc[1:, 1:]
        view3_data = pd.read_csv('../../data/KIRC/data/miRNA1.csv', header=None)
        view3_data = view3_data.T # miRNA
        view3_data = view3_data.iloc[1:, 1:]
        label = pd.read_csv('../../data/KIRC/label.csv', header=None)
        label = label.to_numpy()[1:, 1:]

    print(view1_data.shape)
    print(view2_data.shape)
    print(view3_data.shape)

    if disease == 'tcga_brca':
        view3_data = view3_data.rename(columns={view3_data.columns[0]: "patient_id"})
        view2_data = view2_data.rename(columns={view2_data.columns[0]: "patient_id"})
        view1_data = view1_data.rename(columns={view1_data.columns[0]: "patient_id"})
        label = label.rename(columns={label.columns[0]: "patient_id", label.columns[-1]: "stage"})
        label = label.iloc[:, [0, -1]] # we only need first and last columns, i.e., patient_id and stage

        print(view1_data["patient_id"].nunique(), len(view1_data))
        print(view2_data["patient_id"].nunique(), len(view2_data))
        print(view3_data["patient_id"].nunique(), len(view3_data))
        print(label["patient_id"].nunique(), len(label))

        view1_data = view1_data.drop_duplicates(subset="patient_id")
        view2_data = view2_data.drop_duplicates(subset="patient_id")
        view3_data = view3_data.drop_duplicates(subset="patient_id")
        label = label.drop_duplicates(subset="patient_id")

        dfs = [view1_data, view2_data, view3_data, label]
        dfs = [df.sort_values(by="patient_id").reset_index(drop=True) for df in dfs]
        view1_data, view2_data, view3_data, label = dfs

        view_train_concatenate = view1_data.merge(view3_data, on="patient_id", how="inner").merge(view2_data, on="patient_id",how="inner")

        view_train_concatenate = view_train_concatenate.sort_values(by="patient_id").reset_index(drop=True)

        view1_data, view2_data, view3_data, label, view_train_concatenate = (view1_data.iloc[:-1, 1:],
                                                                             view2_data.iloc[:-1, 1:],
                                                                             view3_data.iloc[:-1, 1:],
                                                                             label.iloc[:-1, 1],
                                                                             view_train_concatenate.iloc[:, 1:])

        dfs2 = [view1_data, view2_data, view3_data, label, view_train_concatenate]
        dfs2 = [df.to_numpy() for df in dfs2]
        view1_data, view2_data, view3_data, label, view_train_concatenate = dfs2


    # Transform features by scaling each feature to a given range (default range is (0, 1))
    scaler = preprocessing.MinMaxScaler()
    view1_data = scaler.fit_transform(view1_data)
    view2_data = scaler.fit_transform(view2_data)
    view3_data = scaler.fit_transform(view3_data)
    view_train_concatenate = np.concatenate((view1_data, view2_data, view3_data), axis=1)

    if disease == 'lihc' or 'kirc' or 'tcga_brca':
        y_true = label
    else:
        y_true = label.to_numpy().flatten()
    print('label dim')
    print(y_true.shape)
    return view1_data, view2_data, view3_data, view_train_concatenate, y_true