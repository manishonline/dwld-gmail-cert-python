import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

import pyro
from pyro.distributions import Normal
from pyro.infer import SVI
from pyro.optim import Adam


N = 100  # size of toy data
p = 1    # number of features

def build_linear_dataset(N, noise_std=0.1):
    X = np.linspace(-6, 6, num=N)
    y = 3 * X + 1 + np.random.normal(0, noise_std, size=N)
    X, y = X.reshape((N, 1)), y.reshape((N, 1))
    X, y = Variable(torch.Tensor(X)), Variable(torch.Tensor(y))
    return torch.cat((X, y), 1)


loss_fn = torch.nn.MSELoss(size_average=False)
optim = torch.optim.Adam(regression_model.parameters(), lr=0.01)
num_iterations = 500

def main():
    data = build_linear_dataset(N, p)
    x_data = data[:, :-1]
    y_data = data[:, -1]
    for j in range(num_iterations):
        # run the model forward on the data
        y_pred = regression_model(x_data)
        # calculate the mse loss
        loss = loss_fn(y_pred, y_data)
        # initialize gradients to zero
        optim.zero_grad()
        # backpropagate
        loss.backward()
        # take a gradient step
        optim.step()
        if (j + 1) % 50 == 0:
            print("[iteration %04d] loss: %.4f" % (j + 1, loss.data[0]))
    # Inspect learned parameters
    print("Learned parameters:")
    for name, param in regression_model.named_parameters():
        print("%s: %.3f" % (name, param.data.numpy()))

if __name__ == '__main__':
    main()
