from torch import optim
import matplotlib.pyplot as plt
import torch

def train(model, loss, loader, epochs):
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for batch in loader:
            batch = batch[0]
            optimizer.zero_grad()
            X, y = batch[:, :-1], batch[:,-2:]
            pred = model(X)
            bloss = loss(pred, y)
            bloss.backward()
            optimizer.step()


def visualize_points(X, pos_idxs, neg_idxs):
    fig, ax = plt.subplots()
    ax.scatter(X[pos_idxs, 0], X[pos_idxs, 1], label="positive")
    ax.scatter(X[neg_idxs, 0], X[neg_idxs, 1], label="negative")
    return ax.legend()


def get_y(w, bias, x_1):
    """
    get the decision boundary point given the weights and x1
    """
    linear_sys = w[0]*x_1
    linear_sys = linear_sys + (torch.ones(x_1.shape)* bias)
    x2 = -linear_sys/w[1]
    return x2


def visualize_decision(weights, bias, X):
    x = torch.linspace(-4, 4, 400)
    y = get_y(weights.squeeze(), bias, x).detach().numpy()
    x = x.numpy()
    n, _ = X.shape
    fig, ax = plt.subplots()
    line1, = ax.plot(x, y, label='decision_boundary')
    line1.set_dashes([2, 2, 10, 2])
    pos_idxs = range(0, int(n / 2))
    ax.scatter(X[pos_idxs, 0], X[pos_idxs, 1], label="positive")
    neg_idxs = range(int(n / 2), n)
    ax.scatter(X[neg_idxs, 0], X[neg_idxs, 1], label="negative")
    ax.legend()
