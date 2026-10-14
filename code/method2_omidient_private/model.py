"""Method 2 + OMIDIENT private: joint D(P,S)->X + VICReg + contrastive + Dirichlet KL.

Shared branch: Method-2-style plain AE encoder (emb=32) + MLP contrastive.
Private branch: OMIDIENT product-Dirichlet encoder (K groups × L=8 → P dim = K·8).
Reconstruction: single joint decoder per modality (no solo shared/specific recon).
"""

import torch
from torch import nn
from torch import lgamma, polygamma
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


def sample_product_dirichlet(h, alpha_layers):
    """OMIDIENT-style product-Dirichlet sample from hidden h (batch × hid).

    Returns:
      em_final: (batch, K * L)
      alphas: (K, batch, L) positive concentration params
    """
    concentration_params = []
    k_groups = len(alpha_layers)
    for k in range(k_groups):
        concentration_params.append(alpha_layers[k](h))

    size = list(concentration_params[0].size())
    x, y = size[0], size[1]
    em_normalized = torch.zeros(x, y, k_groups, device=h.device)
    alphas = torch.stack(concentration_params)
    alphas_pos = []

    for j in range(x):
        for k in range(k_groups):
            alpha_positive = F.sigmoid(alphas[k, j, :])
            alphas_pos.append(alpha_positive)
            u = torch.rand(alpha_positive.shape, device=h.device)
            temp = u.mul(alpha_positive).mul(torch.exp(lgamma(alpha_positive)))
            em = torch.pow(temp, torch.pow(alpha_positive, -1))
            tempreture = torch.pow(em.sum(), -1)
            em = em / em.sum()
            num = em / tempreture
            soft = torch.exp(num) / (torch.exp(num)).sum()
            em_normalized[j, :, k] = soft

    em_final = torch.flatten(em_normalized, start_dim=1)
    alphas_out = torch.stack(alphas_pos).view(k_groups, x, y)
    return em_final, alphas_out


class DirichletPrivateEncoder(nn.Module):
    """Product-Dirichlet private encoder (OMIDIENT private path, encode-only)."""

    def __init__(self, in_dim, units, k_groups):
        super().__init__()
        self.k_groups = k_groups
        self.l1 = nn.Linear(in_dim, units[0])
        self.l2 = nn.Linear(units[0], units[1])
        self.l3 = nn.Linear(units[1], units[2])
        self.alpha_layers = nn.ModuleList([
            nn.Linear(units[2], units[3]) for _ in range(k_groups)
        ])
        self.priv_dim = k_groups * units[3]

    def forward(self, x):
        h = F.tanh(self.l1(x))
        h = F.tanh(self.l2(h))
        h = F.tanh(self.l3(h))
        return sample_product_dirichlet(h, self.alpha_layers)


class SharedAndSpecificEmbedding(nn.Module):
    """Shared AE (Method 2) + Dirichlet private (OMIDIENT) + joint decoder."""

    def __init__(
        self,
        k_groups,
        view_size,
        n_units_shared_1,
        n_units_shared_2,
        n_units_shared_3,
        n_units_priv_1,
        n_units_priv_2,
        n_units_priv_3,
        mlp_size,
    ):
        super().__init__()
        self.k_groups = k_groups
        self.shared_units = {1: n_units_shared_1, 2: n_units_shared_2, 3: n_units_shared_3}
        self.priv_units = {1: n_units_priv_1, 2: n_units_priv_2, 3: n_units_priv_3}

        self.shared_enc = nn.ModuleDict()
        self.specific_enc = nn.ModuleDict()
        self.joint_dec = nn.ModuleDict()
        self.mlp = nn.ModuleDict()

        for i in (1, 2, 3):
            j = str(i)
            u_s = self.shared_units[i]
            u_p = self.priv_units[i]
            vs = view_size[i - 1]
            emb_s = u_s[-1]
            emb_p = k_groups * u_p[-1]

            self.shared_enc[j] = _build_encoder(vs, u_s)
            self.specific_enc[j] = DirichletPrivateEncoder(vs, u_p, k_groups)
            self.joint_dec[j] = _build_decoder(emb_p + emb_s, u_s, vs)
            self.mlp[j] = nn.ModuleList([
                nn.Linear(emb_s, mlp_size[0]),
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
        specific, alphas, shared, shared_mlp = {}, {}, {}, {}
        for i in (1, 2, 3):
            j = str(i)
            specific[i], alphas[i] = self.specific_enc[j](views[i])
            shared[i] = _encode(self.shared_enc[j], views[i])
            mlp_out = F.tanh(self.mlp[j][0](shared[i]))
            shared_mlp[i] = F.tanh(self.mlp[j][1](mlp_out))
        return specific, alphas, shared, shared_mlp

    def decode_joint(self, specific, shared):
        rec = {}
        for i in (1, 2, 3):
            j = str(i)
            rec[i] = _decode(
                self.joint_dec[j], torch.cat([specific[i], shared[i]], dim=1)
            )
        return rec

    def forward(self, view1_input, view2_input, view3_input):
        specific, alphas, shared, shared_mlp = self.encode_views(
            view1_input, view2_input, view3_input
        )
        joint_rec = self.decode_joint(specific, shared)
        return {
            'specific': specific,
            'alphas': alphas,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'joint_rec': joint_rec,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(
        self,
        k_groups=4,
        lambda_rec=0.7,
        lambda_ctr=1.0,
        lambda_vic=1.0,
        lambda_kl=1.0,
        prior_alpha=0.5,
    ):
        super().__init__()
        self.k_groups = k_groups
        self.lambda_rec = lambda_rec
        self.lambda_ctr = lambda_ctr
        self.lambda_vic = lambda_vic
        self.lambda_kl = lambda_kl
        self.prior_alpha = prior_alpha

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
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        return torch.linalg.matrix_norm(rec - ori)

    def _kl_term(self, out):
        """OMIDIENT-style product-Dirichlet KL vs fixed prior alpha."""

        def compute_kl_gamma(alpha_p, alpha_q):
            first_term = lgamma(alpha_p).sum()
            second_term = lgamma(alpha_q).sum(dim=1)
            psi = polygamma(0, alpha_q)
            third_term = (torch.mul((alpha_q - alpha_p), psi)).sum(dim=1)
            return (first_term - second_term + third_term).sum()

        device = out['specific'][1].device
        prior = torch.tensor(self.prior_alpha, device=device)
        kl = 0.0
        for a in (1, 2, 3):
            alphas = out['alphas'][a]  # (K, batch, L)
            for k in range(self.k_groups):
                kl = kl + compute_kl_gamma(prior, alphas[k])
        return kl

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
        L = λ_rec·L_recon + λ_ctr·L_ctr + λ_vic·L_vic + λ_kl·L_kl
        """
        l_recon = self._recon_term(out, ori)
        l_ctr = self._contrastive_term(out, temperature)
        l_vic = self._vicreg_term(out)
        l_kl = self._kl_term(out)
        terms = {
            'l_recon': l_recon,
            'l_ctr': l_ctr,
            'l_vic': l_vic,
            'l_kl': l_kl,
        }
        loss = (
            self.lambda_rec * l_recon
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
            + self.lambda_kl * l_kl
        )
        return loss, terms
