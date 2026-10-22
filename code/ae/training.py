import torch
from torch.autograd import Variable

def train(train_loader, view1_data, view2_data, model, loss_function, temperature, optimizer, epoch, USE_GPU, total_epochs):
    total_loss = 0.0
    total = 0.0
    for iteration_index, train_batch in enumerate(train_loader):
        train_data = train_batch
        d1 = view1_data.shape[1]
        d2 = view2_data.shape[1]
        # Always float32: GPU path used to keep numpy float64 → Double vs Float matmul error
        view1_train_data = torch.as_tensor(train_data[:, :d1], dtype=torch.float32)
        view2_train_data = torch.as_tensor(train_data[:, d1:d1 + d2], dtype=torch.float32)
        view3_train_data = torch.as_tensor(train_data[:, d1 + d2:], dtype=torch.float32)

        if USE_GPU:
            view1_train_data = Variable(view1_train_data.cuda())
            view2_train_data = Variable(view2_train_data.cuda())
            view3_train_data = Variable(view3_train_data.cuda())
        else:
            view1_train_data = Variable(view1_train_data)
            view2_train_data = Variable(view2_train_data)
            view3_train_data = Variable(view3_train_data)

        view1_specific_em, view1_shared_em, view2_specific_em, \
          view2_shared_em, view3_specific_em, \
            view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, \
            view2_shared_rec, view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, \
            view3_shared_mlp = model(view1_train_data, view2_train_data, view3_train_data)


        params = (view1_shared_em, view2_shared_em, view3_shared_em,
                  view1_specific_em, view2_specific_em, view3_specific_em,
                  view1_shared_rec, view2_shared_rec, view3_shared_rec,
                  view1_specific_rec, view2_specific_rec, view3_specific_rec,
                  view1_train_data, view2_train_data, view3_train_data,
                  view1_shared_mlp, view2_shared_mlp, view3_shared_mlp,
                  temperature, view1_shared_em.shape[0])

        loss = loss_function(params)
        optimizer.zero_grad()  # reset optimizer
        loss.backward()  # backpropagation to speard the loss back to the network
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
        d1 = view1_data.shape[1]
        d2 = view2_data.shape[1]
        view1_val_data = torch.as_tensor(val_data[:, :d1], dtype=torch.float32)
        view2_val_data = torch.as_tensor(val_data[:, d1:d1 + d2], dtype=torch.float32)
        view3_val_data = torch.as_tensor(val_data[:, d1 + d2:], dtype=torch.float32)

        if USE_GPU:
            view1_val_data = Variable(view1_val_data.cuda())
            view2_val_data = Variable(view2_val_data.cuda())
            view3_val_data = Variable(view3_val_data.cuda())
        else:
            view1_val_data = Variable(view1_val_data)
            view2_val_data = Variable(view2_val_data)
            view3_val_data = Variable(view3_val_data)

        view1_specific_em, view1_shared_em, view2_specific_em, \
        view2_shared_em, view3_specific_em, \
            view3_shared_em, view1_specific_rec, view1_shared_rec, view2_specific_rec, \
            view2_shared_rec, view3_specific_rec, view3_shared_rec, view1_shared_mlp, view2_shared_mlp, \
            view3_shared_mlp = model(view1_val_data, view2_val_data, view3_val_data)


        params = (view1_shared_em, view2_shared_em, view3_shared_em,
                  view1_specific_em, view2_specific_em, view3_specific_em,
                  view1_shared_rec, view2_shared_rec, view3_shared_rec,
                  view1_specific_rec, view2_specific_rec, view3_specific_rec,
                  view1_val_data, view2_val_data, view3_val_data,
                  view1_shared_mlp, view2_shared_mlp, view3_shared_mlp,
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
