"""SCAN fine-tune: cluster consistency over mined neighbors."""

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def _split(batch, d1, d2, use_gpu):
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    v1 = batch[:, :d1].float()
    v2 = batch[:, d1:d1 + d2].float()
    v3 = batch[:, d1 + d2:].float()
    if use_gpu:
        v1, v2, v3 = v1.cuda(), v2.cuda(), v3.cuda()
    return Variable(v1), Variable(v2), Variable(v3)


def scan_loss(q, neighbor_qs, lambda_ent=2.0, lambda_sharp=1.0):
    """
    q: (B, K)
    neighbor_qs: list of (B, K) soft targets (stopgrad), one per neighbor slot (padded/masked)
    """
    # mean CE to neighbor distributions
    l_cons = 0.0
    n_used = 0
    for nq in neighbor_qs:
        # KL(stopgrad(nq) || q) as consistency
        nq = nq.detach()
        l_cons = l_cons + (nq * (torch.log(nq.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(dim=1).mean()
        n_used += 1
    if n_used == 0:
        l_cons = q.new_zeros(())
    else:
        l_cons = l_cons / n_used

    # entropy of batch-mean q (maximize → minimize negative)
    q_mean = q.mean(dim=0)
    l_ent = (q_mean * torch.log(q_mean.clamp_min(1e-8))).sum()  # negative entropy
    # sharpen: minimize entropy of each q
    l_sharp = (q * torch.log(q.clamp_min(1e-8))).sum(dim=1).mean()

    loss = l_cons + lambda_ent * l_ent + lambda_sharp * l_sharp
    return loss, {'l_cons': l_cons, 'l_ent': l_ent, 'l_sharp': l_sharp}


class EarlyStopper:
    def __init__(self, patience=15, min_delta=0.002):
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
