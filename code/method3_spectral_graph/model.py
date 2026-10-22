"""Method 3 v2: spectral shared + true residual private + view-ID + GRL leak.

v2 private path:
  shared_dec(S) -> X̂_S
  private_dec(P) -> residual  (linear out); target = X - sg(X̂_S)
"""

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


def _decode_sigmoid(layers, z):
    out = z
    for layer in layers[:-1]:
        out = F.tanh(layer(out))
    return torch.sigmoid(layers[-1](out))


def _decode_linear(layers, z):
    out = z
    for layer in layers[:-1]:
        out = F.tanh(layer(out))
    return layers[-1](out)


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
    return var_loss + cov_loss + cross.pow(2).mean()


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
    def __init__(
        self,
        view_size,
        n_units_1,
        n_units_2,
        n_units_3,
        mlp_size,
        spectral_dim=24,
        n_clusters=5,
    ):
        super().__init__()
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}
        self.spectral_dim = spectral_dim
        self.n_clusters = n_clusters
        emb = n_units_1[-1]

        self.shared_enc = nn.ModuleDict()
        self.specific_enc = nn.ModuleDict()
        self.shared_dec = nn.ModuleDict()
        self.private_dec = nn.ModuleDict()
        self.mlp = nn.ModuleDict()

        for i in (1, 2, 3):
            j = str(i)
            u = self.units[i]
            vs = view_size[i - 1]
            e = u[-1]
            self.shared_enc[j] = _build_encoder(vs, u)
            self.specific_enc[j] = _build_encoder(vs, u)
            self.shared_dec[j] = _build_decoder(e, u, vs)
            self.private_dec[j] = _build_decoder(e, u, vs)
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

    def decode_views(self, specific, shared):
        shared_rec, private_res = {}, {}
        for i in (1, 2, 3):
            j = str(i)
            shared_rec[i] = _decode_sigmoid(self.shared_dec[j], shared[i])
            private_res[i] = _decode_linear(self.private_dec[j], specific[i])
        return shared_rec, private_res

    def forward(self, view1, view2, view3, grl_lambda=1.0):
        specific, shared, shared_mlp = self.encode_views(view1, view2, view3)
        shared_rec, private_res = self.decode_views(specific, shared)
        s_bar = (shared[1] + shared[2] + shared[3]) / 3.0
        u_hat = self.f_spec(s_bar)

        view_logits, adv_logits = {}, {}
        for i in (1, 2, 3):
            view_logits[i] = self.view_head(specific[i])
            adv_logits[i] = self.adv_cluster(grad_reverse(specific[i], grl_lambda))

        return {
            'specific': specific,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'shared_rec': shared_rec,
            'private_res': private_res,
            's_bar': s_bar,
            'u_hat': u_hat,
            'view_logits': view_logits,
            'adv_logits': adv_logits,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(
        self,
        lambda_spec=1.0,
        lambda_res=0.3,
        lambda_ctr=0.5,
        lambda_vic=0.1,
        lambda_view=0.5,
        lambda_leak=2.0,
        lambda_srec=0.3,
    ):
        super().__init__()
        self.lambda_spec = lambda_spec
        self.lambda_res = lambda_res
        self.lambda_ctr = lambda_ctr
        self.lambda_vic = lambda_vic
        self.lambda_view = lambda_view
        self.lambda_leak = lambda_leak
        self.lambda_srec = lambda_srec

    @staticmethod
    def contrastive_loss(shared_1, shared_2, temperature, batch_size):
        shared_1 = shared_1 - shared_1.mean()
        shared_2 = shared_2 - shared_2.mean()
        shared_1 = F.normalize(shared_1, p=2, dim=1)
        shared_2 = F.normalize(shared_2, p=2, dim=1)
        return InstanceLoss(batch_size=batch_size, temperature=temperature)(shared_1, shared_2)

    @staticmethod
    def reconstruction_loss(rec, ori):
        rec = rec - rec.mean()
        ori = ori - ori.mean()
        rec = F.normalize(rec, p=2, dim=1)
        ori = F.normalize(ori, p=2, dim=1)
        return torch.linalg.matrix_norm(rec - ori)

    def _contrastive_term(self, out, temperature):
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
        # Match spectral_dim dynamically (v1 r=16 / v2 r=24+)
        u_hat = out['u_hat']
        if u_hat.shape[1] != u_target.shape[1]:
            d = min(u_hat.shape[1], u_target.shape[1])
            l_spec = F.mse_loss(u_hat[:, :d], u_target[:, :d])
        else:
            l_spec = F.mse_loss(u_hat, u_target)

        l_srec = 0.0
        l_res = 0.0
        for a in (1, 2, 3):
            l_srec = l_srec + self.reconstruction_loss(out['shared_rec'][a], ori[a])
            residual_tgt = ori[a] - out['shared_rec'][a].detach()
            l_res = l_res + F.mse_loss(out['private_res'][a], residual_tgt)

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
            'l_srec': l_srec,
            'l_ctr': l_ctr,
            'l_vic': l_vic,
            'l_view': l_view,
            'l_leak': l_leak,
        }
        loss = (
            self.lambda_spec * l_spec
            + self.lambda_res * l_res
            + self.lambda_srec * l_srec
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
            + self.lambda_view * l_view
            + self.lambda_leak * l_leak
        )
        val_score = (
            self.lambda_spec * l_spec
            + self.lambda_res * l_res
            + self.lambda_srec * l_srec
            + self.lambda_ctr * l_ctr
            + self.lambda_vic * l_vic
            + self.lambda_view * l_view
        )
        terms['val_score'] = val_score
        return loss, terms
