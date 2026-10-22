"""Single-optimizer training for Method 3 (spectral + residual private)."""

import torch
from torch.autograd import Variable


def _split_batch_x(batch_x, view1_data, view2_data, use_gpu):
    d1 = view1_data.shape[1]
    d2 = view2_data.shape[1]
    view1 = batch_x[:, :d1]
    view2 = batch_x[:, d1:d1 + d2]
    view3 = batch_x[:, d1 + d2:]
    if use_gpu:
        view1 = Variable(view1.float().cuda())
        view2 = Variable(view2.float().cuda())
        view3 = Variable(view3.float().cuda())
    else:
        view1 = Variable(view1.float())
        view2 = Variable(view2.float())
        view3 = Variable(view3.float())
    return view1, view2, view3


def _to_device(t, use_gpu):
    t = t if torch.is_tensor(t) else torch.as_tensor(t)
    if use_gpu:
        return t.cuda()
    return t


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
    keys = ('l_spec', 'l_res', 'l_srec', 'l_ctr', 'l_vic', 'l_view', 'l_leak', 'val_score')
    sum_terms = {k: 0.0 for k in keys}

    for batch_x, batch_u, batch_y in train_loader:
        view1, view2, view3 = _split_batch_x(batch_x, view1_data, view2_data, use_gpu)
        u_target = Variable(_to_device(batch_u, use_gpu).float())
        cluster_labels = Variable(_to_device(batch_y, use_gpu).long())
        ori = {1: view1, 2: view2, 3: view3}

        out = model(view1, view2, view3, grl_lambda=grl_lambda)
        loss, terms = loss_function(out, ori, u_target, cluster_labels, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = batch_x.shape[0]
        total += n
        total_loss += loss.item()
        for k in sum_terms:
            sum_terms[k] += terms[k].item()

    res = total_loss / total
    print(
        '\n [Epoch: %3d/%3d] Train Loss: %f | spec=%.4f res=%.4f srec=%.4f ctr=%.4f '
        'vic=%.4f view=%.4f leak=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            sum_terms['l_spec'] / total,
            sum_terms['l_res'] / total,
            sum_terms['l_srec'] / total,
            sum_terms['l_ctr'] / total,
            sum_terms['l_vic'] / total,
            sum_terms['l_view'] / total,
            sum_terms['l_leak'] / total,
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
    total_loss = 0.0
    total = 0.0
    keys = ('l_spec', 'l_res', 'l_srec', 'l_ctr', 'l_vic', 'l_view', 'l_leak', 'val_score')
    sum_terms = {k: 0.0 for k in keys}

    with torch.no_grad():
        for batch_x, batch_u, batch_y in val_loader:
            view1, view2, view3 = _split_batch_x(batch_x, view1_data, view2_data, use_gpu)
            u_target = _to_device(batch_u, use_gpu).float()
            cluster_labels = _to_device(batch_y, use_gpu).long()
            ori = {1: view1, 2: view2, 3: view3}
            out = model(view1, view2, view3, grl_lambda=grl_lambda)
            loss, terms = loss_function(out, ori, u_target, cluster_labels, temperature)
            n = batch_x.shape[0]
            total += n
            total_loss += loss.item()
            for k in sum_terms:
                sum_terms[k] += terms[k].item()

    avg_terms = {k: sum_terms[k] / total for k in sum_terms}
    # Checkpoint on constructive val_score (excludes leak)
    res = avg_terms['val_score']
    print(
        '[Epoch: %3d/%3d] Val score: %f | spec=%.4f res=%.4f srec=%.4f ctr=%.4f '
        'vic=%.4f view=%.4f leak=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            avg_terms['l_spec'],
            avg_terms['l_res'],
            avg_terms['l_srec'],
            avg_terms['l_ctr'],
            avg_terms['l_vic'],
            avg_terms['l_view'],
            avg_terms['l_leak'],
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
