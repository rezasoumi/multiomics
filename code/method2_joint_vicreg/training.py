"""Single-optimizer training for Method 2 (joint recon + VICReg + contrastive)."""

import torch
from torch.autograd import Variable


def _split_batch(batch, view1_data, view2_data, use_gpu):
    view1 = torch.tensor(batch[:, :view1_data.shape[1]]).clone().detach()
    view2 = torch.tensor(
        batch[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]]
    ).clone().detach()
    view3 = torch.tensor(
        batch[:, view1_data.shape[1] + view2_data.shape[1]:]
    ).clone().detach()

    if use_gpu:
        view1 = Variable(view1.float().cuda())
        view2 = Variable(view2.float().cuda())
        view3 = Variable(view3.float().cuda())
    else:
        view1 = Variable(view1).type(torch.FloatTensor)
        view2 = Variable(view2).type(torch.FloatTensor)
        view3 = Variable(view3).type(torch.FloatTensor)
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
):
    model.train()
    total_loss = 0.0
    total = 0.0
    sum_terms = {k: 0.0 for k in ('l_recon', 'l_ctr', 'l_vic')}

    for train_batch in train_loader:
        view1, view2, view3 = _split_batch(train_batch, view1_data, view2_data, use_gpu)
        ori = {1: view1, 2: view2, 3: view3}
        out = model(view1, view2, view3)
        loss, terms = loss_function(out, ori, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        n = len(train_batch)
        total += n
        total_loss += loss.item()
        for k in sum_terms:
            sum_terms[k] += terms[k].item()

    res = total_loss / total
    print(
        '\n [Epoch: %3d/%3d] Training Loss: %f | recon=%.4f ctr=%.4f vic=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            sum_terms['l_recon'] / total,
            sum_terms['l_ctr'] / total,
            sum_terms['l_vic'] / total,
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
):
    model.eval()
    total_loss = 0.0
    total = 0.0
    sum_terms = {k: 0.0 for k in ('l_recon', 'l_ctr', 'l_vic')}

    with torch.no_grad():
        for val_batch in val_loader:
            view1, view2, view3 = _split_batch(val_batch, view1_data, view2_data, use_gpu)
            ori = {1: view1, 2: view2, 3: view3}
            out = model(view1, view2, view3)
            loss, terms = loss_function(out, ori, temperature)
            total += len(val_batch)
            total_loss += loss.item()
            for k in sum_terms:
                sum_terms[k] += terms[k].item()

    res = total_loss / total
    avg_terms = {k: sum_terms[k] / total for k in sum_terms}
    print(
        '[Epoch: %3d/%3d] Validation Loss: %f | recon=%.4f ctr=%.4f vic=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            avg_terms['l_recon'],
            avg_terms['l_ctr'],
            avg_terms['l_vic'],
        )
    )
    # All terms positive — val loss is safe for checkpointing
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
