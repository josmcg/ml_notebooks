import torch
from tqdm import tqdm_notebook as prog
from torch import nn
from IPython.display import clear_output

def train(model, loss, loader, optimizer, epochs):
    for epoch in prog(range(epochs)):
        tot_loss = 0.0
        for batch in loader:
            batch = batch[0]
            optimizer.zero_grad()
            X, target = batch[:,:-1], batch[:, -2:]
            pred = model(X)
            ep_loss = loss(pred, target)
            tot_loss += float(ep_loss)
            ep_loss.backward()
            optimizer.step()
    print(f"epoch: {epoch}, loss: {tot_loss}")



class LossWithRegularization(nn.Module):
    def __init__(self, loss, regularizer, lamb=0):
        super(LossWithRegularization, self).__init__()
        self.loss = loss
        self.regularizer = regularizer
        self.lamb = lamb

    def forward(self, pred, gt, weight):
        return self.loss(pred, gt) + (self.lamb * self.regularizer(weight))


def train_regularizer(model, loss, loader, optimizer, epochs):
    for epoch in prog(range(epochs)):
        tot_loss = 0.0
        for batch in loader:
            batch = batch[0]
            optimizer.zero_grad()
            X, target = batch[:, :-1], batch[:, -2:]
            pred = model(X)
            weight = torch.cat([parm.reshape(-1) for parm in model.parameters()])
            ep_loss = loss(pred, target, weight)
            tot_loss += float(ep_loss)
            ep_loss.backward()
            optimizer.step()
    return tot_loss


def cross_validate(w_size, loss, loader, optimizer, epochs, lambs, holdout, lr):
    holdout_loss = nn.MSELoss()
    losses = []
    models = []
    for idx, lamb in enumerate(lambs):
        clear_output(wait=True)
        print(idx)
        loss.lamb = lamb
        model = nn.Linear(w_size, 1, bias=False)
        optim = optimizer(model.parameters(), lr=lr)
        train_regularizer(model, loss, loader, optim, epochs)
        # evaluate model
        hloss = holdout_loss(model(holdout[:, :w_size]), holdout[:, w_size - 1:])
        losses.append(float(hloss))
        models.append(model)
    losses = torch.tensor(losses)
    idx = torch.argmin(losses).item()
    return idx, models[idx]

