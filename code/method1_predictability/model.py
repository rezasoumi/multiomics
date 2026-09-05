"""Method 1: Cross-reconstruction + predictability surrogate (MOCSS base)."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init
from multiomics.code.contrastive_loss import InstanceLoss


def _build_encoder(in_dim, units):
    """units: [h1, h2, h3, emb], e.g. [512, 256, 128, 32]."""
    return nn.ModuleList([
        nn.Linear(in_dim, units[0]),
        nn.Linear(units[0], units[1]),
        nn.Linear(units[1], units[2]),
        nn.Linear(units[2], units[3]),
    ])


def _build_decoder(in_dim, units, out_dim):
    """Mirror MOCSS decoder widths: emb (or 2*emb) -> units[2] -> units[1] -> units[0] -> out."""
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


class SharedAndSpecificEmbedding(nn.Module):
    """
    Per modality: shared encoder S_m, private encoder P_m, MLP projection.
    Predictors: h(P,S), f(S',S''), g(P',P''), k(S).
    """

    def __init__(self, view_size, n_units_1, n_units_2, n_units_3, mlp_size):
        super().__init__()
        self.view_size = list(view_size)
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}
        self.emb_dims = {i: self.units[i][-1] for i in (1, 2, 3)}

        self.shared_enc = nn.ModuleDict()
        self.specific_enc = nn.ModuleDict()
        self.mlp = nn.ModuleDict()
        self.h = nn.ModuleDict()
        self.f = nn.ModuleDict()
        self.g = nn.ModuleDict()
        self.k = nn.ModuleDict()

        for i in (1, 2, 3):
            j = str(i)
            u = self.units[i]
            vs = view_size[i - 1]
            emb = u[-1]

            self.shared_enc[j] = _build_encoder(vs, u)
            self.specific_enc[j] = _build_encoder(vs, u)
            self.mlp[j] = nn.ModuleList([
                nn.Linear(emb, mlp_size[0]),
                nn.Linear(mlp_size[0], mlp_size[1]),
            ])
            # h, f, g take concat of two emb-sized codes
            self.h[j] = _build_decoder(2 * emb, u, vs)
            self.f[j] = _build_decoder(2 * emb, u, vs)
            self.g[j] = _build_decoder(2 * emb, u, vs)
            self.k[j] = _build_decoder(emb, u, vs)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def adversary_parameters(self):
        for net in (self.g, self.k):
            for p in net.parameters():
                yield p

    def main_parameters(self):
        for net in (self.shared_enc, self.specific_enc, self.mlp, self.h, self.f):
            for p in net.parameters():
                yield p

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

    def predict_all(self, specific, shared):
        """Run h, f, g, k for each target modality. Codes may be detached by caller."""
        others = {1: (2, 3), 2: (1, 3), 3: (1, 2)}
        h_rec, f_rec, g_rec, k_rec = {}, {}, {}, {}
        for a in (1, 2, 3):
            j = str(a)
            b, c = others[a]
            h_rec[a] = _decode(self.h[j], torch.cat([specific[a], shared[a]], dim=1))
            f_rec[a] = _decode(self.f[j], torch.cat([shared[b], shared[c]], dim=1))
            g_rec[a] = _decode(self.g[j], torch.cat([specific[b], specific[c]], dim=1))
            k_rec[a] = _decode(self.k[j], shared[a])
        return h_rec, f_rec, g_rec, k_rec

    def forward(self, view1_input, view2_input, view3_input):
        specific, shared, shared_mlp = self.encode_views(view1_input, view2_input, view3_input)
        h_rec, f_rec, g_rec, k_rec = self.predict_all(specific, shared)
        return {
            'specific': specific,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'h_rec': h_rec,
            'f_rec': f_rec,
            'g_rec': g_rec,
            'k_rec': k_rec,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.7, lambda_ctr=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lambda_ctr = lambda_ctr

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

    def _recon_terms(self, out, ori):
        l_std = 0.0
        l_shared = 0.0
        l_adv = 0.0
        l_own = 0.0
        for a in (1, 2, 3):
            l_std = l_std + self.reconstruction_loss(out['h_rec'][a], ori[a])
            l_shared = l_shared + self.reconstruction_loss(out['f_rec'][a], ori[a])
            l_adv = l_adv + self.reconstruction_loss(out['g_rec'][a], ori[a])
            l_own = l_own + self.reconstruction_loss(out['k_rec'][a], ori[a])
        return l_std, l_shared, l_adv, l_own

    def _contrastive_term(self, out, temperature):
        mlp = out['shared_mlp']
        batch_size = mlp[1].shape[0]
        return (
            self.contrastive_loss(mlp[1], mlp[2], temperature, batch_size)
            + self.contrastive_loss(mlp[1], mlp[3], temperature, batch_size)
            + self.contrastive_loss(mlp[2], mlp[3], temperature, batch_size)
        )

    def step1_loss(self, out, ori):
        """Train g and k only: minimize adversarial reconstruction errors."""
        l_std, l_shared, l_adv, l_own = self._recon_terms(out, ori)
        terms = {
            'l_std': l_std,
            'l_shared': l_shared,
            'l_adv': l_adv,
            'l_own': l_own,
            'l_ctr': torch.tensor(0.0, device=l_adv.device),
        }
        loss = l_adv + l_own
        return loss, terms

    def step2_loss(self, out, ori, temperature):
        """Train encoders / h / f: constructive terms minus adversary errors."""
        l_std, l_shared, l_adv, l_own = self._recon_terms(out, ori)
        l_ctr = self._contrastive_term(out, temperature)
        terms = {
            'l_std': l_std,
            'l_shared': l_shared,
            'l_adv': l_adv,
            'l_own': l_own,
            'l_ctr': l_ctr,
        }
        loss = (
            l_std
            + self.alpha * l_shared
            - self.beta * l_adv
            - self.gamma * l_own
            + self.lambda_ctr * l_ctr
        )
        return loss, terms
