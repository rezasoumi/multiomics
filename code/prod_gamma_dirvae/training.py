import torch
from torch.autograd import Variable

def train(train_loader, view1_data, view2_data, model, loss_function, temperature, optimizer, epoch, USE_GPU, total_epochs):
    total_loss = 0.0
    total = 0.0
    for iteration_index, train_batch in enumerate(train_loader):
        train_data = train_batch
        view1_train_data = train_data[:, :view1_data.shape[1]]
        view1_train_data = torch.tensor(view1_train_data).clone().detach()
        view2_train_data = train_data[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]]
        view2_train_data = torch.tensor(view2_train_data).clone().detach()
        view3_train_data = train_data[:, view1_data.shape[1] + view2_data.shape[1]:]
        view3_train_data = torch.tensor(view3_train_data).clone().detach()

        if USE_GPU:
            view1_train_data, view2_train_data, view3_train_data = Variable(view1_train_data.float().cuda()), Variable(
                view2_train_data.float().cuda()), Variable(view3_train_data.float().cuda())
        else:
            view1_train_data = Variable(view1_train_data).type(torch.FloatTensor)
            view2_train_data = Variable(view2_train_data).type(torch.FloatTensor)
            view3_train_data = Variable(view3_train_data).type(torch.FloatTensor)

        view1_specific_em, view1_specific_alpha, view1_shared_em, view2_specific_em, \
            view2_specific_alpha, view2_shared_em, view3_specific_em, view3_specific_alpha, \
            view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, \
            view2_shared_rec, view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, \
            view3_shared_mlp = model(view1_train_data, view2_train_data, view3_train_data)

        prior_alpha = torch.Tensor(1, 8).float().fill_(.5)
        if USE_GPU:
            prior_alpha = prior_alpha.cuda()
        else:
            prior_alpha = prior_alpha.cpu()

        params = (prior_alpha, view1_train_data, view2_train_data, view3_train_data, # For Blie version
                  view1_shared_em, view2_shared_em, view3_shared_em,
                  view1_specific_em, view2_specific_em, view3_specific_em,
                  view1_shared_rec, view2_shared_rec, view3_shared_rec,
                  view1_specific_rec, view2_specific_rec, view3_specific_rec,
                  view1_shared_mlp, view2_shared_mlp, view3_shared_mlp,
                  view1_specific_alpha,
                  view2_specific_alpha,
                  view3_specific_alpha,
                  temperature, view1_shared_em.shape[0])
        # for _ in range(700): # iVON
        #     with optimizer.sampled_params(train=True): # iVON
        loss = loss_function(params)
        optimizer.zero_grad()  # reset optimizer
        loss.backward()  # backpropagation to speard the loss back to the network
        # print('alpha_view_1 grad:')
        # print(model.hid2alpha_specific1.weight.grad)
        # print(model.hid2alpha_specific1.bias.grad)
        # print('******')
        # print('layer1_view_1 grad:')
        # print(model.specific1_l1.weight.grad)
        # print(model.specific1_l1.bias.grad)
        #optimizer.extrapolation() #ExtraAdam
        optimizer.step()  # updating params

        total += len(train_data)
        total_loss += loss.item()
    res = total_loss / total
    print('\n [Epoch: %3d/%3d] Training Loss: %f' % (epoch + 1, total_epochs, res))
    return res


def validation(val_loader, view1_data, view2_data, model, temperature, loss_function, USE_GPU, epoch, total_epochs):
    total_loss = 0.0
    total = 0.0
    for iteration_index, val_batch in enumerate(val_loader):
        val_data = val_batch
        view1_val_data = val_data[:, :view1_data.shape[1]]
        view1_val_data = torch.tensor(view1_val_data).clone().detach()
        view2_val_data = val_data[:, view1_data.shape[1]:view1_data.shape[1] + view2_data.shape[1]]
        view2_val_data = torch.tensor(view2_val_data).clone().detach()
        view3_val_data = val_data[:, view1_data.shape[1] + view2_data.shape[1]:]
        view3_val_data = torch.tensor(view3_val_data).clone().detach()

        if USE_GPU:
            view1_val_data, view2_val_data, view3_val_data = Variable(view1_val_data.cuda()), Variable(
                view2_val_data.cuda()), Variable(view3_val_data.cuda())
        else:
            view1_val_data = Variable(view1_val_data).type(torch.FloatTensor)
            view2_val_data = Variable(view2_val_data).type(torch.FloatTensor)
            view3_val_data = Variable(view3_val_data).type(torch.FloatTensor)

        view1_specific_em, view1_specific_alpha, view1_shared_em, view2_specific_em, \
            view2_specific_alpha, view2_shared_em, view3_specific_em, view3_specific_alpha, \
            view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, \
            view2_shared_rec, view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, \
            view3_shared_mlp = model(view1_val_data, view2_val_data, view3_val_data)

        prior_alpha = torch.Tensor(1, 8).float().fill_(.5)

        params = (prior_alpha, view1_val_data, view2_val_data, view3_val_data,  # For Blie version
                  view1_shared_em, view2_shared_em, view3_shared_em,
                  view1_specific_em, view2_specific_em, view3_specific_em,
                  view1_shared_rec, view2_shared_rec, view3_shared_rec,
                  view1_specific_rec, view2_specific_rec, view3_specific_rec,
                  view1_shared_mlp, view2_shared_mlp, view3_shared_mlp,
                  view1_specific_alpha,
                  view2_specific_alpha,
                  view3_specific_alpha,
                  temperature, view1_shared_em.shape[0])

        loss = loss_function(params)
        total += len(val_data)
        total_loss += loss.item()

    res = total_loss / total
    print('[Epoch: %3d/%3d] Validation Loss: %f' % (epoch + 1, total_epochs, res))

    return res


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
