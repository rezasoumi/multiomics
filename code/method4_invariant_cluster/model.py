"""Method 4: DEC + cross-view agreement on S; residual private + view-ID + GRL leak."""

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


def student_t_q(embeddings, centroids, alpha=1.0):
    """Soft assignment q_ik ∝ (1 + ||z_i - μ_k||^2 / α)^(-(α+1)/2)."""
    # embeddings: (B, D), centroids: (K, D)
    diff = embeddings.unsqueeze(1) - centroids.unsqueeze(0)  # (B, K, D)
    dist_sq = (diff * diff).sum(dim=2)  # (B, K)
    q = (1.0 + dist_sq / alpha).pow(-(alpha + 1.0) / 2.0)
    q = q / q.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return q


def target_distribution(q):
    """DEC target p_ik ∝ q_ik^2 / Σ_i q_ik."""
    weight = (q ** 2) / q.sum(dim=0, keepdim=True).clamp_min(1e-8)
    p = weight / weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return p


def kl_div_pq(p, q):
    return (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=1).mean()


def symmetric_kl(q_a, q_b):
    return 0.5 * (kl_div_pq(q_a, q_b) + kl_div_pq(q_b, q_a))


class SharedAndSpecificEmbedding(nn.Module):
    def __init__(
        self,
        view_size,
        n_units_1,
        n_units_2,
        n_units_3,
        mlp_size,
        n_clusters=5,
        emb_dim=32,
    ):
        super().__init__()
        self.units = {1: n_units_1, 2: n_units_2, 3: n_units_3}
        self.n_clusters = n_clusters
        emb = emb_dim

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

        # Shared DEC centroids (one set for all modalities)
        self.centroids = nn.Parameter(torch.randn(n_clusters, emb) * 0.1)

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
            rec[i] = _decode(
                self.residual_dec[j],
                torch.cat([specific[i], shared[i].detach()], dim=1),
            )
        return rec

    def soft_assignments(self, shared, alpha=1.0):
        q = {}
        for i in (1, 2, 3):
            q[i] = student_t_q(shared[i], self.centroids, alpha=alpha)
        s_bar = (shared[1] + shared[2] + shared[3]) / 3.0
        q_bar = student_t_q(s_bar, self.centroids, alpha=alpha)
        return q, q_bar, s_bar

    def forward(self, view1, view2, view3, grl_lambda=1.0, alpha=1.0):
        specific, shared, shared_mlp = self.encode_views(view1, view2, view3)
        residual_rec = self.decode_residual(specific, shared)
        q, q_bar, s_bar = self.soft_assignments(shared, alpha=alpha)

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
            'q': q,
            'q_bar': q_bar,
            's_bar': s_bar,
            'view_logits': view_logits,
            'adv_logits': adv_logits,
        }


class SharedAndSpecificLoss(nn.Module):
    def __init__(
        self,
        lambda_dec=1.0,
        lambda_agree=1.0,
        lambda_res=0.5,
        lambda_view=0.5,
        lambda_leak=1.0,
        lambda_ctr=0.5,
        lambda_vic=0.1,
        warm_up=False,
    ):
        super().__init__()
        self.lambda_dec = lambda_dec
        self.lambda_agree = lambda_agree
        self.lambda_res = lambda_res
        self.lambda_view = lambda_view
        self.lambda_leak = lambda_leak
        self.lambda_ctr = lambda_ctr
        self.lambda_vic = lambda_vic
        self.warm_up = warm_up  # if True: only ctr (+ optional res), no DEC/agree/leak

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
        if self.lambda_ctr == 0.0:
            return out['shared_mlp'][1].new_zeros(())
        mlp = out['shared_mlp']
        b = mlp[1].shape[0]
        return (
            self.contrastive_loss(mlp[1], mlp[2], temperature, b)
            + self.contrastive_loss(mlp[1], mlp[3], temperature, b)
            + self.contrastive_loss(mlp[2], mlp[3], temperature, b)
        )

    def _vicreg_term(self, out):
        if self.lambda_vic == 0.0:
            return out['shared'][1].new_zeros(())
        l = 0.0
        for a in (1, 2, 3):
            l = l + vicreg_ps_independence(out['specific'][a], out['shared'][a])
        return l

    def forward(self, out, ori, temperature):
        l_res = 0.0
        for a in (1, 2, 3):
            l_res = l_res + self.reconstruction_loss(out['residual_rec'][a], ori[a])

        l_ctr = self._contrastive_term(out, temperature)
        l_vic = self._vicreg_term(out)

        # DEC + agree
        l_dec = out['q'][1].new_zeros(())
        l_agree = out['q'][1].new_zeros(())
        l_leak = out['q'][1].new_zeros(())
        if not self.warm_up:
            for a in (1, 2, 3):
                p = target_distribution(out['q'][a].detach())
                l_dec = l_dec + kl_div_pq(p, out['q'][a])
            l_agree = (
                symmetric_kl(out['q'][1], out['q'][2])
                + symmetric_kl(out['q'][1], out['q'][3])
                + symmetric_kl(out['q'][2], out['q'][3])
            )
            # leak: Adv predicts hard argmax of stopgrad(q_bar)
            hard = out['q_bar'].detach().argmax(dim=1)
            for a in (1, 2, 3):
                l_leak = l_leak + F.cross_entropy(out['adv_logits'][a], hard)

        l_view = 0.0
        for a in (1, 2, 3):
            view_tgt = torch.full(
                (out['view_logits'][a].shape[0],),
                a - 1,
                dtype=torch.long,
                device=out['view_logits'][a].device,
            )
            l_view = l_view + F.cross_entropy(out['view_logits'][a], view_tgt)

        terms = {
            'l_dec': l_dec,
            'l_agree': l_agree,
            'l_res': l_res,
            'l_view': l_view,
            'l_leak': l_leak,
            'l_ctr': l_ctr,
            'l_vic': l_vic,
        }

        if self.warm_up:
            loss = self.lambda_ctr * l_ctr + self.lambda_res * l_res + self.lambda_view * l_view
            val_score = loss
        else:
            loss = (
                self.lambda_dec * l_dec
                + self.lambda_agree * l_agree
                + self.lambda_res * l_res
                + self.lambda_view * l_view
                + self.lambda_leak * l_leak
                + self.lambda_ctr * l_ctr
                + self.lambda_vic * l_vic
            )
            val_score = (
                self.lambda_dec * l_dec
                + self.lambda_agree * l_agree
                + self.lambda_res * l_res
                + self.lambda_view * l_view
                + self.lambda_ctr * l_ctr
                + self.lambda_vic * l_vic
            )
        terms['val_score'] = val_score
        return loss, terms
