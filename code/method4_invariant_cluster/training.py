"""Training for Method 4 (invariant DEC + residual private)."""

import torch
from torch.autograd import Variable


def _split_batch(batch, view1_data, view2_data, use_gpu):
    if not torch.is_tensor(batch):
        batch = torch.as_tensor(batch)
    d1 = view1_data.shape[1]
    d2 = view2_data.shape[1]
    view1 = batch[:, :d1]
    view2 = batch[:, d1:d1 + d2]
    view3 = batch[:, d1 + d2:]
    if use_gpu:
        view1 = Variable(view1.float().cuda())
        view2 = Variable(view2.float().cuda())
        view3 = Variable(view3.float().cuda())
    else:
        view1 = Variable(view1.float())
        view2 = Variable(view2.float())
        view3 = Variable(view3.float())
    return view1, view2, view3


def train(
    train_loader,
    view1_data,
    view2_data,
    model,
    loss_function,
    temperature,
    optimizer,
    epoch,
    use_gpu,
    total_epochs,
    grl_lambda=1.0,
):
    model.train()
    total_loss = 0.0
    total = 0.0
    keys = ('l_dec', 'l_agree', 'l_res', 'l_view', 'l_leak', 'l_ctr', 'l_vic', 'val_score')
    sum_terms = {k: 0.0 for k in keys}

    for train_batch in train_loader:
        view1, view2, view3 = _split_batch(train_batch, view1_data, view2_data, use_gpu)
        ori = {1: view1, 2: view2, 3: view3}
        out = model(view1, view2, view3, grl_lambda=grl_lambda)
        loss, terms = loss_function(out, ori, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = len(train_batch) if not torch.is_tensor(train_batch) else train_batch.shape[0]
        total += n
        total_loss += loss.item()
        for k in sum_terms:
            sum_terms[k] += terms[k].item()

    res = total_loss / total
    print(
        '\n [Epoch: %3d/%3d] Train Loss: %f | dec=%.4f agree=%.4f res=%.4f '
        'view=%.4f leak=%.4f ctr=%.4f vic=%.4f warm=%s'
        % (
            epoch + 1, total_epochs, res,
            sum_terms['l_dec'] / total,
            sum_terms['l_agree'] / total,
            sum_terms['l_res'] / total,
            sum_terms['l_view'] / total,
            sum_terms['l_leak'] / total,
            sum_terms['l_ctr'] / total,
            sum_terms['l_vic'] / total,
            loss_function.warm_up,
        )
    )
    return res


def validation(
    val_loader,
    view1_data,
    view2_data,
    model,
    temperature,
    loss_function,
    use_gpu,
    epoch,
    total_epochs,
    grl_lambda=1.0,
):
    model.eval()
    total = 0.0
    keys = ('l_dec', 'l_agree', 'l_res', 'l_view', 'l_leak', 'l_ctr', 'l_vic', 'val_score')
    sum_terms = {k: 0.0 for k in keys}

    with torch.no_grad():
        for val_batch in val_loader:
            view1, view2, view3 = _split_batch(val_batch, view1_data, view2_data, use_gpu)
            ori = {1: view1, 2: view2, 3: view3}
            out = model(view1, view2, view3, grl_lambda=grl_lambda)
            _, terms = loss_function(out, ori, temperature)
            n = len(val_batch) if not torch.is_tensor(val_batch) else val_batch.shape[0]
            total += n
            for k in sum_terms:
                sum_terms[k] += terms[k].item()

    avg_terms = {k: sum_terms[k] / total for k in sum_terms}
    res = avg_terms['val_score']
    print(
        '[Epoch: %3d/%3d] Val score: %f | dec=%.4f agree=%.4f res=%.4f '
        'view=%.4f leak=%.4f ctr=%.4f vic=%.4f'
        % (
            epoch + 1, total_epochs, res,
            avg_terms['l_dec'], avg_terms['l_agree'], avg_terms['l_res'],
            avg_terms['l_view'], avg_terms['l_leak'], avg_terms['l_ctr'], avg_terms['l_vic'],
        )
    )
    return res, avg_terms


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
def init_centroids_kmeans(model, data_loader, view1_data, view2_data, use_gpu, n_clusters):
    """Initialize DEC centroids from KMeans on mean(S) over a loader pass."""
    from sklearn.cluster import KMeans

    model.eval()
    embs = []
    for batch in data_loader:
        view1, view2, view3 = _split_batch(batch, view1_data, view2_data, use_gpu)
        specific, shared, _ = model.encode_views(view1, view2, view3)
        s_bar = (shared[1] + shared[2] + shared[3]) / 3.0
        embs.append(s_bar.detach().cpu())
    Z = torch.cat(embs, dim=0).numpy()
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=21).fit(Z)
    centers = torch.tensor(km.cluster_centers_, dtype=model.centroids.dtype, device=model.centroids.device)
    model.centroids.data.copy_(centers)
    print('Initialized DEC centroids via KMeans on mean(S) (n={}).'.format(Z.shape[0]))
