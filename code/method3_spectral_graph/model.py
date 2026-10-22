"""Method 3: spectral shared + residual private + view-ID + GRL non-leakage."""

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


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd=1.0):
    return GradReverse.apply(x, lambd)


class SharedAndSpecificEmbedding(nn.Module):
    """
    Shared: Enc -> S; f_spec(mean(S)) -> Û
    Private: Enc -> P; Dec(P, sg(S)) -> X
    Heads: MLP contrastive, view_id(P), Adv(P)->cluster (GRL)
    """

    def __init__(
        self,
        view_size,
        n_units_1,
        n_units_2,
        n_units_3,
        mlp_size,
        spectral_dim=16,
        n_clusters=5,
    ):
        super().__init__()
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}
        self.spectral_dim = spectral_dim
        self.n_clusters = n_clusters
        emb = n_units_1[-1]

        self.shared_enc = nn.ModuleDict()
        self.specific_enc = nn.ModuleDict()
        self.residual_dec = nn.ModuleDict()
        self.mlp = nn.ModuleDict()

        for i in (1, 2, 3):
            j = str(i)
            u = self.units[i]
            vs = view_size[i - 1]
            e = u[-1]
            self.shared_enc[j] = _build_encoder(vs, u)
            self.specific_enc[j] = _build_encoder(vs, u)
            self.residual_dec[j] = _build_decoder(2 * e, u, vs)
            self.mlp[j] = nn.ModuleList([
                nn.Linear(e, mlp_size[0]),
                nn.Linear(mlp_size[0], mlp_size[1]),
            ])

        self.f_spec = nn.Sequential(
            nn.Linear(emb, emb),
            nn.Tanh(),
            nn.Linear(emb, spectral_dim),
        )
        self.view_head = nn.Sequential(
            nn.Linear(emb, emb),
            nn.Tanh(),
            nn.Linear(emb, 3),
        )
        self.adv_cluster = nn.Sequential(
            nn.Linear(emb, emb),
            nn.Tanh(),
            nn.Linear(emb, n_clusters),
        )
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
            h = F.tanh(self.mlp[j][0](shared[i]))
            shared_mlp[i] = F.tanh(self.mlp[j][1](h))
        return specific, shared, shared_mlp

    def decode_residual(self, specific, shared):
        rec = {}
        for i in (1, 2, 3):
            j = str(i)
            s_det = shared[i].detach()
            rec[i] = _decode(self.residual_dec[j], torch.cat([specific[i], s_det], dim=1))
        return rec

    def forward(self, view1, view2, view3, grl_lambda=1.0):
        specific, shared, shared_mlp = self.encode_views(view1, view2, view3)
        residual_rec = self.decode_residual(specific, shared)
        s_bar = (shared[1] + shared[2] + shared[3]) / 3.0
        u_hat = self.f_spec(s_bar)

        view_logits = {}
        adv_logits = {}
        for i in (1, 2, 3):
            view_logits[i] = self.view_head(specific[i])
            adv_logits[i] = self.adv_cluster(grad_reverse(specific[i], grl_lambda))

        return {
            'specific': specific,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'residual_rec': residual_rec,
            's_bar': s_bar,
            'u_hat': u_hat,
            'view_logits': view_logits,
            'adv_logits': adv_logits,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(
        self,
        lambda_spec=1.0,
        lambda_res=0.5,
        lambda_ctr=0.3,
        lambda_vic=0.1,
        lambda_view=0.5,
        lambda_leak=0.5,
    ):
        super().__init__()
        self.lambda_spec = lambda_spec
        self.lambda_res = lambda_res
        self.lambda_ctr = lambda_ctr
        self.lambda_vic = lambda_vic
        self.lambda_view = lambda_view
        self.lambda_leak = lambda_leak

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

    def _contrastive_term(self, out, temperature):
        if self.lambda_ctr == 0.0:
            return out['shared_mlp'][1].new_zeros(())
        mlp = out['shared_mlp']
        batch_size = mlp[1].shape[0]
        return (
            self.contrastive_loss(mlp[1], mlp[2], temperature, batch_size)
            + self.contrastive_loss(mlp[1], mlp[3], temperature, batch_size)
            + self.contrastive_loss(mlp[2], mlp[3], temperature, batch_size)
        )

    def _vicreg_term(self, out):
        if self.lambda_vic == 0.0:
            return out['shared'][1].new_zeros(())
        l_vic = 0.0
        for a in (1, 2, 3):
            l_vic = l_vic + vicreg_ps_independence(out['specific'][a], out['shared'][a])
        return l_vic

    def forward(self, out, ori, u_target, cluster_labels, temperature):
        """
        L = λ_spec L_spec + λ_res L_res + λ_ctr L_ctr + λ_vic L_vic
          + λ_view L_view + λ_leak L_leak
        """
        l_spec = F.mse_loss(out['u_hat'], u_target)

        l_res = 0.0
        for a in (1, 2, 3):
            l_res = l_res + self.reconstruction_loss(out['residual_rec'][a], ori[a])

        l_ctr = self._contrastive_term(out, temperature)
        l_vic = self._vicreg_term(out)

        l_view = 0.0
        l_leak = 0.0
        for a in (1, 2, 3):
            view_tgt = torch.full(
                (out['view_logits'][a].shape[0],),
                a - 1,
                dtype=torch.long,
                device=out['view_logits'][a].device,
            )
            l_view = l_view + F.cross_entropy(out['view_logits'][a], view_tgt)
            l_leak = l_leak + F.cross_entropy(out['adv_logits'][a], cluster_labels)

        terms = {
            'l_spec': l_spec,
            'l_res': l_res,
            'l_ctr': l_ctr,
            'l_vic': l_vic,
            'l_view': l_view,
            'l_leak': l_leak,
        }
        loss = (
            self.lambda_spec * l_spec
            + self.lambda_res * l_res
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
            + self.lambda_view * l_view
            + self.lambda_leak * l_leak
        )
        # Checkpoint score excludes leak (GRL can make scale odd)
        val_score = (
            self.lambda_spec * l_spec
            + self.lambda_res * l_res
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
            + self.lambda_view * l_view
        )
        terms['val_score'] = val_score
        return loss, terms
