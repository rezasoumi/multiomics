"""Training loop for Cluster A."""

import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans


def _split_batch(batch, d1, d2, use_gpu):
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    v1 = batch[:, :d1].float()
    v2 = batch[:, d1:d1 + d2].float()
    v3 = batch[:, d1 + d2:].float()
    if use_gpu:
        v1, v2, v3 = v1.cuda(), v2.cuda(), v3.cuda()
    return Variable(v1), Variable(v2), Variable(v3)


def train_epoch(loader, d1, d2, model, loss_fn, temperature, optimizer, epoch, use_gpu, total_epochs):
    model.train()
    total_loss = 0.0
    total = 0.0
    sums = {k: 0.0 for k in ('l_ctr', 'l_dec', 'l_rec', 'val_score')}
    for batch in loader:
        v1, v2, v3 = _split_batch(batch, d1, d2, use_gpu)
        ori = {1: v1, 2: v2, 3: v3}
        out = model(v1, v2, v3)
        loss, terms = loss_fn(out, ori, temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n = batch.shape[0] if torch.is_tensor(batch) else len(batch)
        total += n
        total_loss += loss.item()
        for k in sums:
            sums[k] += terms[k].item()
    print(
        '\n [Epoch %3d/%3d] Train %f | ctr=%.4f dec=%.4f rec=%.4f warm=%s'
        % (epoch + 1, total_epochs, total_loss / total,
           sums['l_ctr'] / total, sums['l_dec'] / total, sums['l_rec'] / total, loss_fn.warm_up)
    )
    return total_loss / total


def val_epoch(loader, d1, d2, model, loss_fn, temperature, use_gpu, epoch, total_epochs):
    model.eval()
    total = 0.0
    sums = {k: 0.0 for k in ('l_ctr', 'l_dec', 'l_rec', 'val_score')}
    with torch.no_grad():
        for batch in loader:
            v1, v2, v3 = _split_batch(batch, d1, d2, use_gpu)
            ori = {1: v1, 2: v2, 3: v3}
            out = model(v1, v2, v3)
            _, terms = loss_fn(out, ori, temperature)
            n = batch.shape[0] if torch.is_tensor(batch) else len(batch)
            total += n
            for k in sums:
                sums[k] += terms[k].item()
    avg = {k: sums[k] / total for k in sums}
    print(
        '[Epoch %3d/%3d] Val %f | ctr=%.4f dec=%.4f rec=%.4f'
        % (epoch + 1, total_epochs, avg['val_score'], avg['l_ctr'], avg['l_dec'], avg['l_rec'])
    )
    return avg['val_score'], avg


class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.005):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


@torch.no_grad()
def init_centroids_kmeans(model, loader, d1, d2, use_gpu, n_clusters):
    model.eval()
    zs = []
    for batch in loader:
        v1, v2, v3 = _split_batch(batch, d1, d2, use_gpu)
        _, _, z = model.encode(v1, v2, v3)
        zs.append(z.detach().cpu())
    Z = torch.cat(zs, dim=0).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=21).fit(Z)
    centers = torch.tensor(km.cluster_centers_, dtype=model.centroids.dtype, device=model.centroids.device)
    model.centroids.data.copy_(centers)
    print('Initialized DEC centroids from KMeans on Z (n={}).'.format(Z.shape[0]))
