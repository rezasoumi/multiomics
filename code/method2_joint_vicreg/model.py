"""Method 2: joint reconstruction D(P,S)->X + VICReg P⊥S + contrastive (MOCSS base)."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from multiomics.code.contrastive_loss import InstanceLoss


def _build_encoder(in_dim, units):
    return nn.ModuleList([
        nn.Linear(in_dim, units[0]),
        nn.Linear(units[0], units[1]),
        nn.Linear(units[1], units[2]),
        nn.Linear(units[2], units[3]),
    ])


def _build_decoder(in_dim, units, out_dim):
    """Joint decoder: concat(P,S) -> view_size."""
    return nn.ModuleList([
        nn.Linear(in_dim, units[2]),
        nn.Linear(units[2], units[1]),
        nn.Linear(units[1], units[0]),
        nn.Linear(units[0], out_dim),
    ])


def _encode(layers, x):
    out = x
    for layer in layers:
        out = F.tanh(layer(out))
    return out


def _decode(layers, z):
    out = z
    for layer in layers[:-1]:
        out = F.tanh(layer(out))
    return torch.sigmoid(layers[-1](out))


def _off_diagonal(x):
    n = x.shape[0]
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_ps_independence(private, shared, eps=1e-4):
    """Variance hinges + within-cov off-diag + ||Cov(P,S)||_F^2."""
    p = private - private.mean(dim=0)
    s = shared - shared.mean(dim=0)
    batch = max(p.shape[0] - 1, 1)

    std_p = torch.sqrt(p.var(dim=0) + eps)
    std_s = torch.sqrt(s.var(dim=0) + eps)
    var_loss = torch.mean(F.relu(1.0 - std_p)) + torch.mean(F.relu(1.0 - std_s))

    cov_p = (p.T @ p) / batch
    cov_s = (s.T @ s) / batch
    cov_loss = _off_diagonal(cov_p).pow(2).mean() + _off_diagonal(cov_s).pow(2).mean()

    cross = (p.T @ s) / batch
    cross_loss = cross.pow(2).mean()

    return var_loss + cov_loss + cross_loss


class SharedAndSpecificEmbedding(nn.Module):
    """Separate P/S encoders, one joint decoder per modality, MLP on S."""

    def __init__(self, view_size, n_units_1, n_units_2, n_units_3, mlp_size):
        super().__init__()
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}

        self.shared_enc = nn.ModuleDict()
        self.specific_enc = nn.ModuleDict()
        self.joint_dec = nn.ModuleDict()
        self.mlp = nn.ModuleDict()

        for i in (1, 2, 3):
            j = str(i)
            u = self.units[i]
            vs = view_size[i - 1]
            emb = u[-1]

            self.shared_enc[j] = _build_encoder(vs, u)
            self.specific_enc[j] = _build_encoder(vs, u)
            self.joint_dec[j] = _build_decoder(2 * emb, u, vs)
            self.mlp[j] = nn.ModuleList([
                nn.Linear(emb, mlp_size[0]),
                nn.Linear(mlp_size[0], mlp_size[1]),
            ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode_views(self, view1, view2, view3):
        views = {1: view1, 2: view2, 3: view3}
        specific, shared, shared_mlp = {}, {}, {}
        for i in (1, 2, 3):
            j = str(i)
            specific[i] = _encode(self.specific_enc[j], views[i])
            shared[i] = _encode(self.shared_enc[j], views[i])
            mlp_out = F.tanh(self.mlp[j][0](shared[i]))
            shared_mlp[i] = F.tanh(self.mlp[j][1](mlp_out))
        return specific, shared, shared_mlp

    def decode_joint(self, specific, shared):
        rec = {}
        for i in (1, 2, 3):
            j = str(i)
            rec[i] = _decode(self.joint_dec[j], torch.cat([specific[i], shared[i]], dim=1))
        return rec

    def forward(self, view1_input, view2_input, view3_input):
        specific, shared, shared_mlp = self.encode_views(view1_input, view2_input, view3_input)
        joint_rec = self.decode_joint(specific, shared)
        return {
            'specific': specific,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'joint_rec': joint_rec,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, lambda_rec=0.7, lambda_ctr=1.0, lambda_vic=1.0):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_ctr = lambda_ctr
        self.lambda_vic = lambda_vic

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        criterion = InstanceLoss(batch_size=batch_size, temperature=temperature)
        return criterion(shared_1, shared_2)

    @staticmethod
    def reconstruction_loss(rec, ori):
        """MOCSS-style recon for fair comparison."""
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        return torch.linalg.matrix_norm(rec - ori)

    def _recon_term(self, out, ori):
        l_recon = 0.0
        for a in (1, 2, 3):
            l_recon = l_recon + self.reconstruction_loss(out['joint_rec'][a], ori[a])
        return l_recon

    def _contrastive_term(self, out, temperature):
        mlp = out['shared_mlp']
        batch_size = mlp[1].shape[0]
        return (
            self.contrastive_loss(mlp[1], mlp[2], temperature, batch_size)
            + self.contrastive_loss(mlp[1], mlp[3], temperature, batch_size)
            + self.contrastive_loss(mlp[2], mlp[3], temperature, batch_size)
        )

    def _vicreg_term(self, out):
        l_vic = 0.0
        for a in (1, 2, 3):
            l_vic = l_vic + vicreg_ps_independence(out['specific'][a], out['shared'][a])
        return l_vic

    def forward(self, out, ori, temperature):
        """
        L = λ_rec·L_recon + λ_ctr·L_ctr + λ_vic·L_vic
        (no MOCSS orthogonality, no solo shared/specific recon)
        """
        l_recon = self._recon_term(out, ori)
        l_ctr = self._contrastive_term(out, temperature)
        l_vic = self._vicreg_term(out)
        terms = {'l_recon': l_recon, 'l_ctr': l_ctr, 'l_vic': l_vic}
        loss = (
            self.lambda_rec * l_recon
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
        )
        return loss, terms
