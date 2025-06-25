import os
import pickle
import time

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel, PeriodicKernel
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.means import ConstantMean, ZeroMean, LinearMean
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm


def get_cov_kernel(name: str):
    kernel_builders = {'RBF': lambda L: ScaleKernel(RBFKernel(batch_shape=torch.Size([L])), batch_shape=[L]),
                       'Matern32_nu_2_5': lambda L: ScaleKernel(MaternKernel(nu=2.5, batch_shape=[L]), batch_shape=[L]),
                       'Matern32_nu_2_5+Periodic': lambda L: ScaleKernel(MaternKernel(nu=2.5, batch_shape=[L]),
                                                                         batch_shape=[L]) + ScaleKernel(
                           PeriodicKernel(batch_shape=[L]), batch_shape=[L]),
                       'RBFxPeriodic': lambda L: ScaleKernel(RBFKernel(batch_shape=torch.Size([L]))) * ScaleKernel(
                           PeriodicKernel(batch_shape=[L]), batch_shape=[L]),
                       'RBF+Periodic': lambda L: ScaleKernel(RBFKernel(batch_shape=torch.Size([L]))) + ScaleKernel(
                           PeriodicKernel(batch_shape=[L]), batch_shape=[L]),
                       'RBFxPeriodic+Matern_nu_2_5': lambda L: ScaleKernel(
                           RBFKernel(batch_shape=torch.Size([L])) * PeriodicKernel(batch_shape=[L]),
                           batch_shape=[L]) + ScaleKernel(
                           MaternKernel(nu=2.5, batch_shape=[L]), batch_shape=[L]), }
    if name in kernel_builders:
        return kernel_builders[name]
    else:
        raise ValueError(f"Unknown kernel name: {name}. Available kernels are: {list(kernel_builders.keys())}")


def get_mean_kernel(name: str, input_size):
    mean_builders = {'Constant': lambda L: ConstantMean(batch_shape=torch.Size([L])),
                     'Zero': lambda L: ZeroMean(batch_shape=[L]),
                     'Linear': lambda L: LinearMean(input_size=input_size, batch_shape=[L]), }
    if name in mean_builders:
        return mean_builders[name]
    else:
        raise ValueError(f"Unknown mean name: {name}. Available means are: {list(mean_builders.keys())}")


class GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, num_tasks, num_latents, mean_module, covar_module):
        """
        inducing_points: Tensor (M, D)
        num_tasks:       # of outputs, here 24
        num_latents:     # of latent GPs to mix (<= num_tasks)
        mean_module:     a gpytorch.means.Mean object, batched [num_latents]
        covar_module:    a gpytorch.kernels.Kernel within ScaleKernel, batched
        """
        # 1) variational distribution for each latent GP
        M = inducing_points.size(0)
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(M,
                                                                                        batch_shape=torch.Size(
                                                                                            [num_latents]))
        # 2) single‐task variational strategy, batched
        base_vs = gpytorch.variational.VariationalStrategy(self,
                                                           inducing_points.unsqueeze(0).expand(num_latents, M, -1),
                                                           variational_distribution,
                                                           learn_inducing_locations=True)
        # 3) mix latents → tasks
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(base_vs, num_tasks, num_latents,
                                                                           latent_dim=-1)
        super().__init__(variational_strategy)

        # plug in chosen mean & covar modules
        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        # x: (..., D)
        mean = self.mean_module(x)  # → (num_latents, N)
        covar = self.covar_module(x)  # → (num_latents, N, N)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


def create_model(best_config, inducing_points, input_size, horizon=24, device="cpu"):
    # Extract necessary parameters from best_config
    num_inducing = best_config['num_inducing']
    num_latents = best_config['num_latents']  # Compression dimension, max = horizon (24 values in this case)
    mean_kernel = get_mean_kernel(name=best_config.get('mean_kernel'), input_size=input_size)(num_latents)
    cov_kernel = get_cov_kernel(name=best_config.get('cov_kernel'))(num_latents)
    # Create the Approximate Gaussian Process model and Gaussian likelihood
    model = GPModel(inducing_points=inducing_points, num_tasks=horizon,
                    # Falls fehler aufkommt. Model war vorher num_tasks=y_train.shape[1]
                    num_latents=num_latents, mean_module=mean_kernel, covar_module=cov_kernel).to(device)
    likelihood = MultitaskGaussianLikelihood(num_tasks=horizon).to(device)
    return model, likelihood


def load_model(best_config, model_save_filename, likelihood_save_filename, X_train, input_size, device):
    num_inducing = best_config.get('num_inducing')
    perm = torch.randperm(X_train.size(0))
    inducing_points = X_train[perm[:num_inducing]].clone()

    model, likelihood = create_model(best_config, inducing_points, input_size=input_size)

    # Load the saved state dictionaries
    # map_location handles loading to the correct device
    model.load_state_dict(torch.load(model_save_filename, map_location=device))
    likelihood.load_state_dict(torch.load(likelihood_save_filename, map_location=device))
    print("Model and likelihood loaded successfully.")

    # Ensure model and likelihood are on the correct device and in eval mode
    model = model.to(device)
    likelihood = likelihood.to(device)
    model.eval()
    likelihood.eval()

    return model, likelihood


def load_file(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def train_and_eval(X_tr, y_tr, X_val, y_val, kernel_builder, mean_builder, num_inducing, num_latents, lr, num_epochs,
                   GPModel, device, fold_i=None, outer_pbar=None):
    """
    Trains a variational Gaussian Process model using GPyTorch and evaluates it on a validation set.

    This function builds a batched variational GP model with the specified kernel and mean functions,
    trains it using the ELBO objective, and computes the RMSE on the validation set.

    Args:
        X_tr (torch.Tensor): Training input features of shape (n_train, input_dim).
        y_tr (torch.Tensor): Training targets of shape (n_train, output_dim).
        X_val (torch.Tensor): Validation input features of shape (n_val, input_dim).
        y_val (torch.Tensor): Validation targets of shape (n_val, output_dim).
        kernel_builder (Callable[[int], gpytorch.kernels.Kernel]): Function that returns a batched kernel module.
        mean_builder (Callable[[int], gpytorch.means.Mean]): Function that returns a batched mean module.
        num_inducing (int): Number of inducing points for the variational GP.
        num_latents (int): Number of latent GPs in the variational model.
        lr (float): Learning rate for the Adam optimizer.
        num_epochs (int): Number of training epochs.
        device (torch.device): Device to run the training on (e.g., 'cuda' or 'cpu').

    Returns:
        float: Root Mean Squared Error (RMSE) of the model predictions on the validation set.
    """
    # slice inducing points
    perm = torch.randperm(X_tr.size(0))
    inducing = X_tr[perm[:num_inducing]].clone()

    # build mean & covar modules (batched over num_latents)
    mean_mod = mean_builder(num_latents).to(device)
    covar_mod = kernel_builder(num_latents).to(device)

    # instantiate model + likelihood
    model = GPModel(inducing, y_tr.shape[1], num_latents, mean_mod, covar_mod).to(device)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=y_tr.shape[1]).to(device)

    # optim + ELBO
    optimizer = torch.optim.Adam([{'params': model.parameters()}, {'params': likelihood.parameters()}], lr=lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=X_tr.size(0))

    model.train()
    likelihood.train()

    train_ds = torch.utils.data.TensorDataset(X_tr, y_tr)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=512, shuffle=True)
    for epoch in range(num_epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if outer_pbar is not None:
            outer_pbar.set_description(f"Fold {fold_i + 1} | Epoch {epoch + 1}/{num_epochs}")
            outer_pbar.refresh()

    # eval
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        pred = likelihood(model(X_val)).mean.cpu()
    rmse = torch.sqrt(((pred - y_val.cpu()) ** 2).mean()).item()
    return rmse


def perform_gridsearch(cov_kernels, mean_kernels, inducing_list, latents_list, epochs_list, lr_list, device,
                       RANDOM_SEED, X_train, y_train, GPModel):
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    best = {'rmse': float('inf')}

    total_it = len(cov_kernels) * len(mean_kernels) * len(inducing_list) * len(latents_list) * len(lr_list) * len(
        epochs_list)
    curr_iteration = 1
    # Grid Search through all parameters
    for cov_kernel_name in cov_kernels:
        kbuilder = get_cov_kernel(cov_kernel_name)
        for mean_kernel_name in mean_kernels:
            mbuilder = get_mean_kernel(mean_kernel_name, input_size=X_train.shape[1])
            for num_inducing in inducing_list:
                for num_latents in latents_list:
                    for lr in lr_list:
                        for epochs in epochs_list:
                            start_time = time.time()
                            rmses = []
                            folds = list(kf.split(X_train))
                            fold_pbar = tqdm(total=len(folds),
                                             desc=f"[{curr_iteration}/{total_it}] {cov_kernel_name}-{mean_kernel_name}",
                                             position=0)
                            for fold_i, (tr_idx, val_idx) in enumerate(folds):
                                X_tr, X_val = X_train[tr_idx], X_train[val_idx]
                                y_tr, y_val = y_train[tr_idx], y_train[val_idx]
                                rmse = train_and_eval(X_tr, y_tr, X_val, y_val, kbuilder, mbuilder, num_inducing,
                                                      num_latents, lr, epochs, GPModel, device, fold_i=fold_i,
                                                      outer_pbar=fold_pbar)
                                fold_pbar.set_postfix(rmse=rmse)
                                rmses.append(rmse)
                            fold_pbar.close()
                            # Average rmse across folds
                            avg_rmse = np.mean(rmses)
                            print(
                                f'{curr_iteration:3d}/{total_it} in {time.time() - start_time:5.2f}s | GP({mean_kernel_name}, {cov_kernel_name}) | '
                                f'num_inducing={num_inducing}, num_latents={num_latents}, '
                                f'lr={lr:.4f}, epochs={epochs} → '
                                f'RMSE={avg_rmse:.4f}')
                            curr_iteration += 1
                            if avg_rmse < best['rmse']:
                                best.update(
                                    {'rmse': avg_rmse, 'cov_kernel': cov_kernel_name, 'mean_kernel': mean_kernel_name,
                                     'num_inducing': num_inducing, 'num_latents': num_latents, 'lr': lr,
                                     'epochs': epochs})
    return best


def save_training_results(model, likelihood, train_losses, test_losses, name_addition='', model_folder: str = 'models'):
    with open(os.path.join(model_folder, f'train_losses{name_addition}.pkl'), 'wb') as f:
        pickle.dump(train_losses, f)
    with open(os.path.join(model_folder, f'test_losses{name_addition}.pkl'), 'wb') as f:
        pickle.dump(test_losses, f)
    torch.save(model.state_dict(), os.path.join(model_folder, f'model_state_dict{name_addition}.pth'))
    torch.save(likelihood.state_dict(), os.path.join(model_folder, f'likelihood_state_dict{name_addition}.pth'))
    print("Saved model and likelihood successfully.")
