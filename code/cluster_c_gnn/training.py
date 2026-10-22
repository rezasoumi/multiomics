"""Training helpers for Cluster C (GCN + DEC)."""

import torch
from sklearn.cluster import KMeans

from multiomics.code.cluster_c_gnn.model import target_distribution, kl_div_pq


class EarlyStopper:
    def __init__(self, patience=30, min_delta=0.002):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float('inf')

    def early_stop(self, val):
        if val < self.best:
            self.best = val
            self.counter = 0
        elif val > self.best + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@torch.no_grad()
def init_centroids(model, x, a_hat, train_idx, n_clusters, use_gpu, seed=21):
    model.eval()
    out = model(x, a_hat)
    z = out['z'].detach().cpu().numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=seed).fit(z[train_idx])
    centers = torch.tensor(km.cluster_centers_, dtype=model.centroids.dtype)
    if use_gpu:
        centers = centers.cuda()
    model.centroids.data.copy_(centers)


def train_epoch(model, x, a_hat, train_idx, optimizer, sharpen=2.5):
    model.train()
    out = model(x, a_hat)
    p = target_distribution(out['q'].detach(), sharpen=sharpen)
    loss = kl_div_pq(p[train_idx], out['q'][train_idx])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


@torch.no_grad()
def val_loss(model, x, a_hat, val_idx, sharpen=2.5):
    model.eval()
    out = model(x, a_hat)
    p = target_distribution(out['q'], sharpen=sharpen)
    return float(kl_div_pq(p[val_idx], out['q'][val_idx]).item())
