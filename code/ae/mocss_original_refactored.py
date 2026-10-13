import numpy as np
from multiomics.code.loading_data import load_data
from torch import nn
from multiomics.code import util
import torch.nn.functional as F
import torch.optim
import torch.nn.init as init
from multiomics.code.contrastive_loss import InstanceLoss
import argparse
from sklearn.cluster import KMeans
import multiomics.code.evaluation as evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

lasso = False

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SharedAndSpecificLoss(nn.Module):
    def __init__(self):
        super(SharedAndSpecificLoss, self).__init__()

    @staticmethod
    def orthogonal_loss(shared, specific):
        shared = shared - shared.mean()
        specific = specific - specific.mean()
        shared = F.normalize(shared, p=2, dim=1)
        specific = F.normalize(specific, p=2, dim=1)
        # This is not inner product but they use it because the get better result
        correlation_matrix = torch.mul(shared, specific)
        cost = correlation_matrix.mean()
        return cost

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        assert (shared_1.dim() == 2)
        assert (shared_2.dim() == 2)
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        criterion_instance = InstanceLoss(batch_size=batch_size, temperature=temperature)
        loss = criterion_instance(shared_1, shared_2)
        return loss

    @staticmethod
    def reconstruction_loss(rec, ori):
        assert (rec.dim() == 2)
        assert (ori.dim() == 2)
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        # This is eucleadian norm: sqrt(sum(a_ij^2))
        loss = torch.linalg.matrix_norm(rec-ori)
        return loss

    @staticmethod
    def l1_loss(em):
        l1_batch = torch.norm(em, 1, dim=1)
        return l1_batch.mean()

    def forward(self, params):
        (shared1_output, shared2_output, shared3_output, specific1_output, specific2_output, specific3_output,
        shared1_rec, shared2_rec, shared3_rec, specific1_rec, specific2_rec, specific3_rec, ori1, ori2, ori3,
        shared1_mlp, shared2_mlp, shared3_mlp, temperature, batch_size) = params

        # orthogonal restrict
        orthogonal_loss1 = self.orthogonal_loss(shared1_output, specific1_output)
        orthogonal_loss2 = self.orthogonal_loss(shared2_output, specific2_output)
        orthogonal_loss3 = self.orthogonal_loss(shared3_output, specific3_output)
        orthogonal_loss_all = orthogonal_loss1 + orthogonal_loss2 + orthogonal_loss3

        # Contrastive Loss
        contrastive_loss1 = self.contrastive_loss(shared1_mlp, shared2_mlp, temperature, batch_size)
        contrastive_loss2 = self.contrastive_loss(shared1_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss3 = self.contrastive_loss(shared2_mlp, shared3_mlp, temperature, batch_size)
        contrastive_loss_all = contrastive_loss1 + contrastive_loss2 + contrastive_loss3

        # reconstruction Loss
        reconst_loss1 = self.reconstruction_loss(shared1_rec, ori1) + self.reconstruction_loss(specific1_rec, ori1)
        reconst_loss2 = self.reconstruction_loss(shared2_rec, ori2) + self.reconstruction_loss(specific2_rec, ori2)
        reconst_loss3 = self.reconstruction_loss(shared3_rec, ori3) + self.reconstruction_loss(specific3_rec, ori3)
        reconstruction_loss_all = reconst_loss1 + reconst_loss2 + reconst_loss3

        # lasso loss
        l1_loss1 = self.l1_loss(specific1_output)
        l1_loss2 = self.l1_loss(specific2_output)
        l1_loss3 = self.l1_loss(specific3_output)
        l1_loss_all = l1_loss1 + l1_loss2 + l1_loss3

        if lasso:
            loss_total = orthogonal_loss_all + contrastive_loss_all + .7 * reconstruction_loss_all + l1_loss_all
        else:
            loss_total = orthogonal_loss_all + contrastive_loss_all + .7 * reconstruction_loss_all

        return loss_total


class SharedAndSpecificEmbedding(nn.Module):
    def __init__(self, view_size, n_units_1, n_units_2, n_units_3, mlp_size):
        super(SharedAndSpecificEmbedding, self).__init__()

        units = {'1': n_units_1, '2': n_units_2, '3': n_units_3}
        # This dictionary is used to store NN-lineartransformations,
        # where each key follows this pattern [shared/specific]+[omic_view]+'[encoder/decoder_layers]'
        linears = {}
        # Views are i in [1, 2, 3] for three omics data
        for i in range(1,4):
            j = str(i)
            for type in ['shared', 'specific']:
                # encoder (4 layers)
                linears[type+j+'_l1'] = nn.Linear(view_size[i-1], units[j][0])
                linears[type+j+'_l2'] = nn.Linear(units[j][0], units[j][1])
                linears[type+j+'_l3'] = nn.Linear(units[j][1], units[j][2])
                linears[type+j+'_l4'] = nn.Linear(units[j][2], units[j][3])
                # decoder  (4 layers)
                linears[type+j+'_l3_'] = nn.Linear(units[j][3], units[j][2])
                linears[type+j+'_l2_'] = nn.Linear(units[j][2], units[j][1])
                linears[type+j+'_l1_'] = nn.Linear(units[j][1], units[j][0])
                linears[type+j+'_rec'] = nn.Linear(units[j][0], view_size[i-1])

            linears['view'+j+'_mlp1'] = nn.Linear(units[j][3], mlp_size[0])
            linears['view'+j+'_mlp2'] = nn.Linear(mlp_size[0], mlp_size[1])

        # The order is very important since _modules is an OrderedDictionary
        self.shared1_l1, self.shared1_l2, self.shared1_l3, self.shared1_l4, self.shared1_l3_, self.shared1_l2_, \
            self.shared1_l1_, self.shared1_rec, self.specific1_l1, self.specific1_l2, self.specific1_l3, self.specific1_l4,\
            self.specific1_l3_, self.specific1_l2_, self.specific1_l1_, self.specific1_rec, \
            self.shared2_l1, self.shared2_l2, self.shared2_l3, self.shared2_l4, self.shared2_l3_, self.shared2_l2_, \
            self.shared2_l1_, self.shared2_rec, self.specific2_l1, self.specific2_l2, self.specific2_l3, self.specific2_l4,\
            self.specific2_l3_, self.specific2_l2_, self.specific2_l1_, self.specific2_rec, \
            self.shared3_l1, self.shared3_l2, self.shared3_l3, self.shared3_l4, self.shared3_l3_, self.shared3_l2_, \
            self.shared3_l1_, self.shared3_rec, self.specific3_l1, self.specific3_l2, self.specific3_l3, self.specific3_l4,\
            self.specific3_l3_, self.specific3_l2_, self.specific3_l1_, self.specific3_rec, self.view1_mlp1, self.view1_mlp2,\
            self.view2_mlp1, self.view2_mlp2, self.view3_mlp1, self.view3_mlp2 \
            = linears['shared1_l1'], linears['shared1_l2'], linears['shared1_l3'], linears['shared1_l4'], \
              linears['shared1_l3_'], linears['shared1_l2_'], linears['shared1_l1_'], linears['shared1_rec'], \
              linears['specific1_l1'], linears['specific1_l2'], linears['specific1_l3'], linears['specific1_l4'],\
              linears['specific1_l3_'], linears['specific1_l2_'], linears['specific1_l1_'], linears['specific1_rec'], \
              linears['shared2_l1'], linears['shared2_l2'], linears['shared2_l3'], linears['shared2_l4'], \
              linears['shared2_l3_'], linears['shared2_l2_'], linears['shared2_l1_'], linears['shared2_rec'], \
              linears['specific2_l1'], linears['specific2_l2'], linears['specific2_l3'], linears['specific2_l4'],\
              linears['specific2_l3_'], linears['specific2_l2_'], linears['specific2_l1_'], linears['specific2_rec'], \
              linears['shared3_l1'], linears['shared3_l2'], linears['shared3_l3'], linears['shared3_l4'], \
              linears['shared3_l3_'], linears['shared3_l2_'], linears['shared3_l1_'], linears['shared3_rec'], \
              linears['specific3_l1'], linears['specific3_l2'], linears['specific3_l3'], linears['specific3_l4'], \
              linears['specific3_l3_'], linears['specific3_l2_'], linears['specific3_l1_'], linears['specific3_rec'], \
              linears['view1_mlp1'], linears['view1_mlp2'], linears['view2_mlp1'], linears['view2_mlp2'], \
              linears['view3_mlp1'], linears['view3_mlp2']

        def init_weights(s):
            original = s.__dict__
            d = s.__dict__.get('_modules')
            for k, v in d.items():
                init.kaiming_normal_(v.weight)
            # items() creates a new view object and therefore we need to update the original dictionary
            original.update({'_modules': d})

        # Init weight
        init_weights(self)

    def forward(self, view1_input, view2_input, view3_input):
        def encoder_decoder(layers, input_data):
            l1, l2, l3, l4, l3_,l2_, l1_, rec = layers
            # We give batch_size x original_features_dim
            out = F.tanh(l1(input_data))
            out = F.tanh(l2(out))
            out = F.tanh(l3(out))
            # We get batch_size x embedding_dim
            em = F.tanh(l4(out))
            out = F.tanh(l3_(em))
            out = F.tanh(l2_(out))
            out = F.tanh(l1_(out))
            # We get batch_size x original_features_dim
            reconstructed = torch.sigmoid(rec(out))
            return em, reconstructed

        # View1
        l1_specific = (self.specific1_l1, self.specific1_l2, self.specific1_l3, self.specific1_l4, self.specific1_l3_,
                       self.specific1_l2_, self.specific1_l1_, self.specific1_rec)

        view1_specific_em, view1_specific_rec = encoder_decoder(l1_specific, view1_input)
        l1_shared= (self.shared1_l1, self.shared1_l2, self.shared1_l3, self.shared1_l4, self.shared1_l3_,
                    self.shared1_l2_, self.shared1_l1_, self.shared1_rec)
        view1_shared_em, view1_shared_rec = encoder_decoder(l1_shared, view1_input)
        view1_shared_mlp = F.tanh(self.view1_mlp1(view1_shared_em))
        view1_shared_mlp = F.tanh(self.view1_mlp2(view1_shared_mlp))

        # View2
        l2_specific = (self.specific2_l1, self.specific2_l2, self.specific2_l3, self.specific2_l4, self.specific2_l3_,
                       self.specific2_l2_, self.specific2_l1_, self.specific2_rec)
        view2_specific_em, view2_specific_rec = encoder_decoder(l2_specific, view2_input)
        l2_shared= (self.shared2_l1, self.shared2_l2, self.shared2_l3, self.shared2_l4, self.shared2_l3_,
                    self.shared2_l2_, self.shared2_l1_, self.shared2_rec)
        view2_shared_em, view2_shared_rec = encoder_decoder(l2_shared, view2_input)
        view2_shared_mlp = F.tanh(self.view2_mlp1(view2_shared_em))
        view2_shared_mlp = F.tanh(self.view2_mlp2(view2_shared_mlp))

        # View3
        l3_specific = (self.specific3_l1, self.specific3_l2, self.specific3_l3, self.specific3_l4, self.specific3_l3_,
                       self.specific3_l2_, self.specific3_l1_, self.specific3_rec)
        view3_specific_em, view3_specific_rec = encoder_decoder(l3_specific, view3_input)
        l3_shared = (self.shared3_l1, self.shared3_l2, self.shared3_l3, self.shared3_l4, self.shared3_l3_,
                     self.shared3_l2_, self.shared3_l1_, self.shared3_rec)
        view3_shared_em, view3_shared_rec = encoder_decoder(l3_shared, view3_input)
        view3_shared_mlp = F.tanh(self.view3_mlp1(view3_shared_em))
        view3_shared_mlp = F.tanh(self.view3_mlp2(view3_shared_mlp))

        return view1_specific_em, view1_shared_em, view2_specific_em, view2_shared_em, view3_specific_em, \
               view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, view2_shared_rec, \
               view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, view3_shared_mlp


def main(args):
    method = "AutoEncoder"
    disease = 'brca' #'kirc' 'coad' 'lihc'
    num_clust = {'lihc': 2, 'coad': 4, 'kirc':2, 'brca':5}[disease]


    view1_data, view2_data, view3_data, view_train_concatenate, y_true = load_data(disease)

    # Build Model
    model = SharedAndSpecificEmbedding(
            view_size=[view1_data.shape[1], view2_data.shape[1], view3_data.shape[1]],
            n_units_1=[512, 256, 128, 32], n_units_2=[512, 256, 128, 32],
            n_units_3=[256, 128, 64, 32], mlp_size=[32, 8]
        )

    # Load test data
    ls = [{'loss': 100000000, 'config': 'test'}]
    path = ('../../results/models_'+disease+'_ProdGammaDirVae')
    import os
    for (dir_path, dir_names, file_names) in os.walk(path):
        for config in dir_names:
            name = os.path.join(config, 'loss.npy')
            f = os.path.join(path, name)
            if os.path.exists(f):
                l = np.load(f)
                dict = {'loss': l, 'config': config}
                ls = np.append(ls, dict)
    loss_min = min(ls, key=lambda x: x['loss'])
    folder = loss_min['config']
    all_params = {
        'brca': {'vae': '0.0006_0.0004', 'ProdGammaDirVae': '0.0003_0.0005_4', 'ae': '0.0004_0.0007',
                 'GammaDirVae': '0.0003_0.0007', 'lapdirvae': '0.0006_0.0007'},
        'lihc': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0003_0.0006', 'lapdirvae': '0.0002_0.0005',
                 'ProdGammaDirVae': '0.0005_0.0007_4', 'vae': '0.0005_0.0007'},
        'kirc': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0001_0.0006', 'lapdirvae': '0.0002_0.0005',
                 'ProdGammaDirVae': '0.0003_0.0005_4', 'vae': '0.0003_0.0007'},
        'coad': {'ae': '0.0002_0.0007', 'GammaDirVae': '0.0001_0.0006', 'lapdirvae': '0.0001_0.0006',
                 'ProdGammaDirVae': '0.0002_0.0003_5', 'vae': '0.0002_0.0006'}}
    folder = all_params[disease]['ProdGammaDirVae']
    desired_path = os.path.join(path, folder)
    data = np.load(desired_path + '/test_data_{}.npy'.format(disease))
    label = np.load(desired_path + '/test_label_{}.npy'.format(disease), allow_pickle=True)

    print(folder)

    # Load model
    ls2 = [{'loss': 100000000, 'config': 'test'}]
    if lasso == True:
        path2 = "../../results/models_" + disease + "_lassoAE"
    else:
        path2 = "../../results/models_"+disease+"_ae"
    import os
    for (dir_path, dir_names, file_names) in os.walk(path2):
        for config in dir_names:
            name = os.path.join(config, 'loss.npy')
            f = os.path.join(path2, name)
            if os.path.exists(f):
                l = np.load(f)
                dict = {'loss': l, 'config': config}
                ls2 = np.append(ls2, dict)
    loss_min2 = min(ls2, key=lambda x: x['loss'])
    folder2 = loss_min2['config']
    folder2 = all_params[disease]['ae']
    model_path = os.path.join(path2, folder2)
    desired_path = model_path    
    model.load_state_dict(torch.load(model_path + '/model_{}'.format(disease)))
    model.eval()

    print(folder2)

    setup_seed(2)

    # get the result
    view1_test_data = data[:, :view1_data.shape[1]]
    view1_test_data = torch.tensor(view1_test_data, dtype=torch.float32).clone().detach()
    view2_test_data = data[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]]
    view2_test_data = torch.tensor(view2_test_data, dtype=torch.float32).clone().detach()
    view3_test_data = data[:, view1_data.shape[1] + view2_data.shape[1]:]
    view3_test_data = torch.tensor(view3_test_data, dtype=torch.float32).clone().detach()

    view1_specific_em_new, view1_shared_em_new, view2_specific_em_new, \
         view2_shared_em_new, view3_specific_em_new,  \
        view3_shared_em_new, view1_specific_rec_new, view1_shared_rec_new, view2_specific_rec_new, \
        view2_shared_rec_new, view3_specific_rec_new, view3_shared_rec_new, view1_shared_mlp_new, view2_shared_mlp_new, \
        view3_shared_mlp_new = model(view1_test_data, view2_test_data, view3_test_data)
    view_shared_common = (view1_shared_em_new + view2_shared_em_new + view3_shared_em_new) / 3
    final_embedding = torch.cat(
        (view1_specific_em_new, view2_specific_em_new, view3_specific_em_new, view_shared_common), dim=1)
    final_embedding = final_embedding.detach().numpy()

    if disease == 'coad':
        truth = label.flatten().astype('int')
    elif disease == 'lihc':
        lst = label[:, 0].flatten()
        unique_vals = list(set(lst))  # Find unique values
        mapping = {val: idx for idx, val in enumerate(unique_vals)}  # Assign unique numbers
        truth_stage = [mapping[val] for val in lst]
        truth_class = label[:, 1].flatten().astype('int')
    elif disease == 'kirc':
        truth = label[:, 1].flatten().astype('int')
    else:
        truth = label.flatten()

    if disease == 'lihc':
        util.plot_with_path(data, truth_class, desired_path + "/data", method)
        util.plot_with_path(final_embedding, truth_class, desired_path + "/final_em", method)
        best_inertia = float("inf")
        best_labels = None
        for i in range(30):
            kmeans = KMeans(n_clusters=num_clust, init='k-means++', random_state=i)
            labels = kmeans.fit_predict(final_embedding)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth_class, best_labels)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))

        X_train, X_test, y_train, y_test = train_test_split(final_embedding, truth_class, test_size=0.25, random_state=12)

        knn = KNeighborsClassifier(n_neighbors=num_clust)

        # Train the model
        knn.fit(X_train, y_train)

        # Predict on test set
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"kNN acc: {accuracy:.2f}")

        util.plot_with_path(data, truth_stage, desired_path + "/data", method)
        util.plot_with_path(final_embedding, truth_stage, desired_path + "/final_em", method)
        best_inertia = float("inf")
        best_labels = None
        for i in range(30):
            kmeans = KMeans(n_clusters=num_clust, init='k-means++', random_state=i)
            labels = kmeans.fit_predict(final_embedding)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_labels = labels
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(np.asarray(truth_stage), best_labels)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))

        X_train, X_test, y_train, y_test = train_test_split(final_embedding, np.asarray(truth_stage), test_size=0.25, random_state=12)

        knn = KNeighborsClassifier(n_neighbors=num_clust)

        # Train the model
        knn.fit(X_train, y_train)

        # Predict on test set
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"kNN acc: {accuracy:.2f}")

    else:

        util.plot_with_path(final_embedding, truth, model_path + "/final_em", method)
        best_inertia = float('inf')
        y_pred = None
        for i in range(30):
            kmeans = KMeans(n_clusters=num_clust, init='k-means++', random_state=i)
            labels = kmeans.fit_predict(final_embedding)
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                y_pred = labels
        nmi_, ari_, f_score_, acc_, v_, ch = evaluation.evaluate(truth, y_pred)
        print('\n' + ' ' * 8 + '|==>  nmi: %.4f,  ari: %.4f,  f_score: %.4f,  acc: %.4f,  v_measure: %.4f,  '
                               'ch_index: %.4f  <==|' % (nmi_, ari_, f_score_, acc_, v_, ch))

        X_train, X_test, y_train, y_test = train_test_split(final_embedding, truth, test_size=0.25, random_state=12)
        knn = KNeighborsClassifier(n_neighbors=num_clust)

        # Train the model
        knn.fit(X_train, y_train)

        # Predict on test set
        y_pred = knn.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        print(f"kNN acc: {accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Method Running')
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')

    main(args)
