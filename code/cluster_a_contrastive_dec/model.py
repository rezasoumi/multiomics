"""Cluster A: Enc_m → h_m → Z=mean(h); contrastive on MLP(h); DEC on Z; optional recon from Z."""

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


def student_t_q(z, centroids, alpha=1.0):
    diff = z.unsqueeze(1) - centroids.unsqueeze(0)
    dist_sq = (diff * diff).sum(dim=2)
    q = (1.0 + dist_sq / alpha).pow(-(alpha + 1.0) / 2.0)
    return q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)


def target_distribution(q, sharpen=2.5):
    weight = (q ** sharpen) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    return weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)


def kl_div_pq(p, q):
    return (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=1).mean()


class MultiViewClusterModel(nn.Module):
    def __init__(self, view_size, n_units_1, n_units_2, n_units_3, mlp_size, n_clusters=5):
        super().__init__()
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}
        emb = n_units_1[-1]
        self.n_clusters = n_clusters

        self.enc = nn.ModuleDict()
        self.dec = nn.ModuleDict()
        self.mlp = nn.ModuleDict()
        for i in (1, 2, 3):
            j = str(i)
            u = self.units[i]
            vs = view_size[i - 1]
            e = u[-1]
            self.enc[j] = _build_encoder(vs, u)
            self.dec[j] = _build_decoder(emb, u, vs)  # decode from fused Z dim
            self.mlp[j] = nn.ModuleList([
                nn.Linear(e, mlp_size[0]),
                nn.Linear(mlp_size[0], mlp_size[1]),
            ])

        self.centroids = nn.Parameter(torch.randn(n_clusters, emb) * 0.1)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def encode(self, view1, view2, view3):
        views = {1: view1, 2: view2, 3: view3}
        h, p = {}, {}
        for i in (1, 2, 3):
            j = str(i)
            h[i] = _encode(self.enc[j], views[i])
            t = F.tanh(self.mlp[j][0](h[i]))
            p[i] = F.tanh(self.mlp[j][1](t))
        z = (h[1] + h[2] + h[3]) / 3.0
        return h, p, z

    def decode_from_z(self, z):
        rec = {}
        for i in (1, 2, 3):
            rec[i] = _decode(self.dec[str(i)], z)
        return rec

    def forward(self, view1, view2, view3, alpha=1.0):
        h, p, z = self.encode(view1, view2, view3)
        rec = self.decode_from_z(z)
        q = student_t_q(z, self.centroids, alpha=alpha)
        return {'h': h, 'p': p, 'z': z, 'rec': rec, 'q': q}


class ClusterALoss(nn.Module):
    def __init__(self, lambda_ctr=1.0, lambda_dec=1.0, lambda_rec=0.0, sharpen=2.5, warm_up=False):
        super().__init__()
        self.lambda_ctr = lambda_ctr
        self.lambda_dec = lambda_dec
        self.lambda_rec = lambda_rec
        self.sharpen = sharpen
        self.warm_up = warm_up

    @staticmethod
    def contrastive_loss(a, b, temperature, batch_size):
        a = a - a.mean()
        b = b - b.mean()
        a = F.normalize(a, p=2, dim=1)
        b = F.normalize(b, p=2, dim=1)
        return InstanceLoss(batch_size=batch_size, temperature=temperature)(a, b)

    @staticmethod
    def reconstruction_loss(rec, ori):
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        return torch.linalg.matrix_norm(rec - ori)

    def forward(self, out, ori, temperature):
        p = out['p']
        b = p[1].shape[0]
        l_ctr = (
            self.contrastive_loss(p[1], p[2], temperature, b)
            + self.contrastive_loss(p[1], p[3], temperature, b)
            + self.contrastive_loss(p[2], p[3], temperature, b)
        )

        l_rec = out['z'].new_zeros(())
        if self.lambda_rec != 0.0:
            for a in (1, 2, 3):
                l_rec = l_rec + self.reconstruction_loss(out['rec'][a], ori[a])

        l_dec = out['z'].new_zeros(())
        if not self.warm_up:
            tgt = target_distribution(out['q'].detach(), sharpen=self.sharpen)
            l_dec = kl_div_pq(tgt, out['q'])

        terms = {'l_ctr': l_ctr, 'l_dec': l_dec, 'l_rec': l_rec}
        if self.warm_up:
            loss = self.lambda_ctr * l_ctr + self.lambda_rec * l_rec
        else:
            loss = (
                self.lambda_ctr * l_ctr
                + self.lambda_dec * l_dec
                + self.lambda_rec * l_rec
            )
        terms['val_score'] = loss
        return loss, terms
