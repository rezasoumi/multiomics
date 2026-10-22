"""Pure PyTorch 2-layer GCN + DEC head (no PyG dependency)."""

import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init


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


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, x, a_hat):
        return self.lin(torch.matmul(a_hat, x))


class PatientGCN(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=32, n_clusters=5):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hidden)
        self.gcn2 = GCNLayer(hidden, out_dim)
        self.centroids = nn.Parameter(torch.randn(n_clusters, out_dim) * 0.1)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, a_hat, alpha=1.0):
        h = F.relu(self.gcn1(x, a_hat))
        z = self.gcn2(h, a_hat)
        q = student_t_q(z, self.centroids, alpha=alpha)
        return {'z': z, 'q': q}
