"""Two-step training for Method 1 predictability surrogate."""

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


def _detach_codes(specific, shared):
    return (
        {i: specific[i].detach() for i in specific},
        {i: shared[i].detach() for i in shared},
    )


def _set_requires_grad(params, flag):
    for p in params:
        p.requires_grad_(flag)


def train(
    train_loader,
    view1_data,
    view2_data,
    model,
    loss_function,
    temperature,
    optimizer_main,
    optimizer_adv,
    epoch,
    use_gpu,
    total_epochs,
):
    model.train()
    total_loss = 0.0
    total = 0.0
    sum_terms = {k: 0.0 for k in ('l_std', 'l_shared', 'l_adv', 'l_own', 'l_ctr')}

    adv_params = list(model.adversary_parameters())

    for train_batch in train_loader:
        view1, view2, view3 = _split_batch(train_batch, view1_data, view2_data, use_gpu)
        ori = {1: view1, 2: view2, 3: view3}

        # Encode once
        specific, shared, shared_mlp = model.encode_views(view1, view2, view3)

        # ---- Step 1: update g, k only (stop-grad on P, S) ----
        specific_d, shared_d = _detach_codes(specific, shared)
        h_rec, f_rec, g_rec, k_rec = model.predict_all(specific_d, shared_d)
        out_step1 = {
            'specific': specific_d,
            'shared': shared_d,
            'shared_mlp': shared_mlp,
            'h_rec': h_rec,
            'f_rec': f_rec,
            'g_rec': g_rec,
            'k_rec': k_rec,
        }
        loss1, _ = loss_function.step1_loss(out_step1, ori)
        optimizer_adv.zero_grad()
        loss1.backward()
        optimizer_adv.step()

        # ---- Step 2: update encoders / h / f / mlp (freeze g, k weights) ----
        _set_requires_grad(adv_params, False)
        # Re-encode so graphs are fresh after step-1 optimizer update
        specific, shared, shared_mlp = model.encode_views(view1, view2, view3)
        h_rec, f_rec, g_rec, k_rec = model.predict_all(specific, shared)
        out_step2 = {
            'specific': specific,
            'shared': shared,
            'shared_mlp': shared_mlp,
            'h_rec': h_rec,
            'f_rec': f_rec,
            'g_rec': g_rec,
            'k_rec': k_rec,
        }
        loss2, terms = loss_function.step2_loss(out_step2, ori, temperature)
        optimizer_main.zero_grad()
        loss2.backward()
        optimizer_main.step()
        _set_requires_grad(adv_params, True)

        n = len(train_batch)
        total += n
        total_loss += loss2.item()
        for k in sum_terms:
            sum_terms[k] += terms[k].item()

    res = total_loss / total
    print(
        '\n [Epoch: %3d/%3d] Training Loss: %f | std=%.4f shared=%.4f adv=%.4f own=%.4f ctr=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            sum_terms['l_std'] / total,
            sum_terms['l_shared'] / total,
            sum_terms['l_adv'] / total,
            sum_terms['l_own'] / total,
            sum_terms['l_ctr'] / total,
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
    sum_terms = {k: 0.0 for k in ('l_std', 'l_shared', 'l_adv', 'l_own', 'l_ctr')}

    with torch.no_grad():
        for val_batch in val_loader:
            view1, view2, view3 = _split_batch(val_batch, view1_data, view2_data, use_gpu)
            ori = {1: view1, 2: view2, 3: view3}
            out = model(view1, view2, view3)
            loss2, terms = loss_function.step2_loss(out, ori, temperature)
            total += len(val_batch)
            total_loss += loss2.item()
            for k in sum_terms:
                sum_terms[k] += terms[k].item()

    res = total_loss / total
    avg_terms = {k: sum_terms[k] / total for k in sum_terms}
    # Constructive score (no -adv/-own): safer for checkpointing than signed L_step2
    constructive = (
        avg_terms['l_std']
        + loss_function.alpha * avg_terms['l_shared']
        + loss_function.lambda_ctr * avg_terms['l_ctr']
    )
    print(
        '[Epoch: %3d/%3d] Validation Loss: %f | constructive=%.4f std=%.4f shared=%.4f '
        'adv=%.4f own=%.4f ctr=%.4f'
        % (
            epoch + 1,
            total_epochs,
            res,
            constructive,
            avg_terms['l_std'],
            avg_terms['l_shared'],
            avg_terms['l_adv'],
            avg_terms['l_own'],
            avg_terms['l_ctr'],
        )
    )
    return res, constructive, avg_terms


class EarlyStopper:
    def __init__(self, patience=30, min_delta=1000):
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
