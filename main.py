import logging
import os
import time
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from opacus import PrivacyEngine
from opacus.accountants.utils import get_noise_multiplier
from opacus.utils.batch_memory_manager import BatchMemoryManager
from pytorch_lightning.callbacks import EarlyStopping
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from commons import DATA_PATH, FCNN, data_split
from lazy_utils import extract_grad, recover_tensors

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def private_train(model, train_loader, optimizer, device, epochs, batch_size):
    model.train()
    criterion = torch.nn.MSELoss()
    for l in range(epochs):
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=batch_size, optimizer=optimizer) as new_data_loader:
            # for i_batch, sampled_batch in enumerate(new_data_loader):
            for _, (data, target) in enumerate(new_data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data.float())
                loss = criterion(output.float(), target.float().unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


def main(
    data_name,
    epoch=10,
    tot_trial=15,
    alpha=0.1,
    lam=10,
    iseed=2023,
    train_size=300,
    hidden_width=64,
    p_sim=16,
    batch_size=10,
):
    hidden_layer = [hidden_width, hidden_width, ]
    results = pd.DataFrame(
        columns=['itrial', 'dataset', 'method', 'coverage', 'width', 'time'])

    ###########################################################################################
    # Prepare Data
    np.random.seed(iseed)
    if data_name == 'sim_normal':
        p = p_sim
        X = np.random.normal(0, 5.0, size=(5000, p))
        Y = np.zeros(5000)
        beta = np.random.beta(1.0, 2.5, size=p)
        for i in range(X.shape[0]):
            Y[i] = np.sqrt(max(0, np.inner(X[i], beta)))

        Y = Y + np.random.normal(0, 0.2, size=5000)  # change the noise level

    if data_name == 'blog':
        blog_data = np.loadtxt(os.path.join(DATA_PATH, 'blogData_train.csv'), delimiter=',')
        X = blog_data[:, :-1]
        Y = np.log(1+blog_data[:, -1])
        p = X.shape[1]

    if data_name == 'med':
        meps_data = np.loadtxt(os.path.join(DATA_PATH, 'meps_data.txt'))
        X = meps_data[:, :-1]
        Y = meps_data[:, -1]
        p = X.shape[1]

    if data_name == "crime":
        # remove predictors with missing values
        communities_data = np.loadtxt(os.path.join(DATA_PATH, 'communities_data.txt')).astype(float)
        X = communities_data[:, :-1]
        Y = communities_data[:, -1]
        p = X.shape[1]
    ###########################################################################################

    def nn_fit(X, y, hidden_layer=[128,]):
        n, p = X.shape
        this_nn = FCNN(p, hidden_layer, 1)
        early_stopping = EarlyStopping('val_loss', mode='min')
        trainer = pl.Trainer(
            callbacks=[early_stopping], max_steps=10000, max_epochs=epoch,
            enable_progress_bar=False,
            # accelerator='gpu', devices=1,
        )
        train_loader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32).view(-1, 1)
            ), batch_size=batch_size, num_workers=20)
        #! Here I change the batch size to 1 to be consistent with the privacy training
        trainer.fit(this_nn, train_loader, train_loader)
        return this_nn

    def nn_fit_predict(X, y, X_test, hidden_layer=[128,]):
        this_nn = nn_fit(X, y, hidden_layer)
        with torch.no_grad():
            full_pred_test = this_nn(torch.tensor(
                X_test, dtype=torch.float32)).squeeze().numpy()
        return full_pred_test

    ###########################################################################################
    # Compute PIs functions

    def lazy_predict(grads, flat_params, shape_info, full_nn, X_train, y_train, X_test, lam=100.0, hidden_layer=(128, ), ):
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        _, p = X_train.shape
        full_pred_train = full_nn(X_train)
        lazy = Ridge(alpha=float(lam)).fit(grads, y_train -
                                           full_pred_train.detach().numpy())
        delta = lazy.coef_
        lazy_retrain_params = torch.FloatTensor(delta) + flat_params
        lazy_retrain_Tlist = recover_tensors(
            lazy_retrain_params.reshape(-1), shape_info)
        lazy_retrain_nn = FCNN(p, hidden_layer, 1, activation=nn.Sigmoid)
        # hidden_layers probably doesn't need to be an argument here - get it from the structure
        for k, param in enumerate(lazy_retrain_nn.parameters()):
            param.data = lazy_retrain_Tlist[k]
        # lazy_pred_train = lazy_retrain_nn(X_train)
        with torch.no_grad():
            lazy_pred_test = lazy_retrain_nn(X_test).squeeze().numpy()
        return lazy_pred_test

    def compute_PIs_lazy(X, Y, X1, alpha, full_nn=None, lam=100.0, hidden_layer=hidden_layer):
        n = len(X)
        n1 = len(X1)
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n, n1))
        warnings.filterwarnings("ignore")
        for i in tqdm(range(n), desc='lazyPI'):
            X_train_change = np.delete(X, i, 0)
            Y_train_change = np.delete(Y, i)
            # TODO optimize
            # with warnings.catch_warnings():
            grads, flat_params, shape_info = extract_grad(
                torch.tensor(X_train_change, dtype=torch.float32), full_nn)

            muh_vals_LOO = lazy_predict(grads, flat_params, shape_info, full_nn, X_train_change, Y_train_change, np.r_[X[i].reshape((1, -1)), X1],
                                        lam=float(lam), hidden_layer=hidden_layer)
            # print(muh_vals_LOO)
            resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
            muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
        ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)

        return pd.DataFrame(
            np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO, axis=1).T[-ind_q],
                  np.sort(muh_LOO_vals_testpoint.T + resids_LOO, axis=1).T[ind_q-1]],
            columns=['lower', 'upper'])

    def compute_PIs_jacknife_plus(X, Y, X1, alpha, fit_muh_fun, hidden_layer):
        n = len(X)
        n1 = len(X1)
        resids_LOO = np.zeros(n)
        muh_LOO_vals_testpoint = np.zeros((n, n1))
        for i in tqdm(range(n), desc='jacknife'):
            muh_vals_LOO = fit_muh_fun(np.delete(X, i, 0), np.delete(Y, i),
                                       np.r_[X[i].reshape((1, -1)), X1],
                                       hidden_layer)
            resids_LOO[i] = np.abs(Y[i] - muh_vals_LOO[0])
            muh_LOO_vals_testpoint[i] = muh_vals_LOO[1:]
        ind_q = (np.ceil((1-alpha)*(n+1))).astype(int)
        ###############################
        # construct prediction intervals
        ###############################

        return pd.DataFrame(
            np.c_[np.sort(muh_LOO_vals_testpoint.T - resids_LOO, axis=1).T[-ind_q],
                  np.sort(muh_LOO_vals_testpoint.T + resids_LOO, axis=1).T[ind_q-1]],
            columns=['lower', 'upper'])

    # target_delta = c/(train_size**1.2)
    # target_epsilon = c/(train_size**1.2)
    target_epsilon = 0.01
    target_delta = 0.001

    noise_multiplier = get_noise_multiplier(
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        sample_rate=batch_size/train_size,
        epochs=1,
        accountant='gdp',
    )

    for itrial in tqdm(range(tot_trial), desc='Total Trail'):
        try:
            print(f'trial # {itrial}')
            np.random.seed(202301 + itrial)
            [X_train, X_test, Y_train, Y_test] = data_split(X, Y, train_size)

            train_loader = DataLoader(
                torch.utils.data.TensorDataset(
                    torch.tensor(X_train, dtype=torch.float32),
                    torch.tensor(Y_train, dtype=torch.float32).view(-1, 1)
                ), batch_size=1, num_workers=20)
            start_time = time.time()
            PIs = compute_PIs_jacknife_plus(
                X_train, Y_train, X_test, alpha, nn_fit_predict, hidden_layer)
            coverage = ((PIs['lower'] <= Y_test) & (PIs['upper'] >= Y_test)).mean()
            width = (PIs['upper'] - PIs['lower']).mean()
            results.loc[len(results)] = [itrial, data_name, 'jacknife+',
                                         coverage, width, time.time() - start_time]

            # logging.info('Lazy with finetuned full nn')
            start_time = time.time()
            full_nn = nn_fit(X_train, Y_train, hidden_layer=hidden_layer)

            PIs = compute_PIs_lazy(X_train, Y_train, X_test,
                                   alpha, full_nn=full_nn, lam=float(lam), hidden_layer=hidden_layer)
            coverage = ((PIs['lower'] <= Y_test) & (PIs['upper'] >= Y_test)).mean()
            width = (PIs['upper'] - PIs['lower']).mean()
            results.loc[len(results)] = [itrial, data_name, 'lazy_finetune',
                                         coverage, width, time.time() - start_time]

            model = FCNN(input_dim=X_train.shape[1],
                         hidden_widths=hidden_layer, output_dim=1)

            optimizer = model.configure_optimizers()
            privacy_engine = PrivacyEngine(accountant='gdp')

            start_time = time.time()
            private_model, private_optimizer, dp_data_loader = privacy_engine.make_private(
                module=model,
                optimizer=optimizer,
                data_loader=train_loader,
                noise_multiplier=noise_multiplier,
                max_grad_norm=2.0,
                poisson_sampling=False,
                # clipping=True,
            )

            private_train(private_model, dp_data_loader, private_optimizer, "cpu", epochs=1, batch_size=batch_size)

            PIs = compute_PIs_lazy(X_train, Y_train, X_test, alpha,
                                   full_nn=private_model, lam=float(lam), hidden_layer=hidden_layer)
            coverage = ((PIs['lower'] <= Y_test) & (PIs['upper'] >= Y_test)).mean()
            width = (PIs['upper'] - PIs['lower']).mean()
            results.loc[len(results)] = [itrial, data_name, 'lazy_privacy',
                                         coverage, width, time.time() - start_time]
            # dp_results = pd.DataFrame(['itrial', 'dataset', 'coverage',
            #             'theory_coverage', 'noise_multiplier', 'epsilon', 'delta', 'loss'])

            results.to_csv(
                f'SGD_{data_name}_p{p}_hidden{hidden_width}_seed{iseed}_lam{lam}_e{epoch}_batch{batch_size}_{itrial}.csv', index=False)
            print('Results saved in ' +
                  f'SGD_{data_name}_p{p}_hidden{hidden_width}_seed{iseed}_lam{lam}_e{epoch}_batch{batch_size}_{itrial}.csv')
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            print('Error', e)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', '-dn', type=str, default='sim_normal')
    parser.add_argument('--epoch', '-e', default=10, type=int)
    parser.add_argument('--tot_trial', '-t', default=15, type=int,
                        help='total number of trials')
    parser.add_argument('--alpha', '-a', default=0.1, type=float)
    parser.add_argument('--train_size', '-n', help='training size', default=300, type=int)
    parser.add_argument('--hidden_width', '-w', default=64, type=int)
    parser.add_argument('--lam', '-l', default=10, type=int,
                        help='penalty parameter for ridge')
    parser.add_argument('--iseed', '-s', default=2023,
                        help='data generating random seed')
    parser.add_argument('--psim', '-p', default=16, type=int,
                        help='simulation p')
    # parser.add_argument('--dp_constant', '-c', default=0.01, type=float,
    #                     help='target delta/epsilon constant')
    # parser.add_argument('--noise', '-ns', default=2.0, type=float,
    #                     help='noise multiplier for the privacy training')
    args = parser.parse_args()
    main(
        args.data_name,
        epoch=args.epoch,
        tot_trial=args.tot_trial,
        alpha=args.alpha,
        lam=args.lam,
        iseed=args.iseed,
        train_size=args.train_size,
        hidden_width=args.hidden_width,
        p_sim=args.psim,
    )
