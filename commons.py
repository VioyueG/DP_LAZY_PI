import time

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader

DATA_PATH = './data'


class FCNN(pl.LightningModule):
    """
     Creates a fully connected neural network
    :param input_dim: int, dimension of input data
    :param hidden_widths: list of size of hidden widths of network
    :param output_dim: dimension of output
    :param activation: activation function for hidden layers (defaults to ReLU)
    :return: A network
     """

    def __init__(self,
                 input_dim: int,
                 hidden_widths: list,
                 output_dim: int,
                 activation=nn.ReLU):
        super().__init__()
        structure = [input_dim] + list(hidden_widths) + [output_dim]
        layers = []
        for j in range(len(structure) - 1):
            act = activation if j < len(structure) - 2 else nn.Identity
            layers += [nn.Linear(structure[j], structure[j + 1]), act()]

        self.net = nn.Sequential(
            *layers
        )

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x).float()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        x, y = batch
        y_hat = self.net(x)
        loss = nn.MSELoss()(y_hat, y)
        # Logging to TensorBoard by default
        self.log("my_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_nb):
        x, y = batch
        loss = nn.MSELoss()(self.net(x), y)
        self.log('val_loss', loss)
        return loss


def data_split(X_full, Y_full, train_size):
    '''
    Sample n=200 data from the original dataset to be used as training data
    X_full: Full X data matrix (for communities/meps/blog)
    Y_full: Full Y data matrix (for communities/meps/blog)
    Return: A list of data matrices of train/test(or validation) pairs
    '''
    n_tot = len(Y_full)
    idx = np.random.choice(n_tot, train_size, replace=False)
    not_idx = [i not in idx for i in range(n_tot)]
    X_train = X_full[idx, :]
    X_test = X_full[not_idx, :]
    Y_train = Y_full[idx, ]
    Y_test = Y_full[not_idx, ]
    return ([X_train, X_test, Y_train, Y_test])


def nn_fit_predict(X, y, X_test, hidden_widths=[128,], tol=1e-4):
    n, p = X.shape
    full_nn = FCNN(p, hidden_widths, 1)
    early_stopping = EarlyStopping('val_loss', min_delta=tol)
    trainer = pl.Trainer(callbacks=[early_stopping], max_steps=200)
    train_loader = DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32).view(-1, 1)
        ), batch_size=n)
    t0 = time.time()
    # with io.capture_output() as captured:
    trainer.fit(full_nn, train_loader, train_loader)
    with torch.no_grad():
        full_pred_test = full_nn(torch.tensor(X_test, dtype=torch.float32)).squeeze().numpy()
    return full_pred_test
