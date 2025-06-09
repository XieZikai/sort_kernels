import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import os, subprocess
import argparse

import botorch
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.acquisition import ExpectedImprovement
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel
from copy import deepcopy


"""
该策略为hybrid策略：
1. 训练时不右乘；
2. 预测（EI）时使用average EI
"""


class MergeKernel(Kernel):
    has_lengthscale = True
    def forward(self, X, X2, **params):
        if len(X.shape) > 2:
            kernel_mat = torch.sum((X - X2)**2, axis=-1)
        else:
            kernel_mat = torch.sum((X[:, None, :] - X2)**2, axis=-1)
        return torch.exp(-self.lengthscale * kernel_mat)


def merge_sort(arr):
    # print(f'Handling array: {arr}')
    if len(arr) == 1:
        return []
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = deepcopy(arr[:mid])
        right_half = deepcopy(arr[mid:])

        left_feature = merge_sort(left_half)
        right_feature = merge_sort(right_half)

        # print(f'L & R: {left_feature}, {right_feature}')
        feature = [] + left_feature + right_feature

        i = j = k = 0

        # Merge the two halves into the original list
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                arr[k] = left_half[i]
                i += 1
                feature.append(-1)
            else:
                arr[k] = right_half[j]
                j += 1
                feature.append(1)
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1
            feature.append(1)

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1
            feature.append(-1)

        # print('Return: ', feature)
        return feature[:-1]


def featurize(x, anchors):
    """
    Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    features = []
    x_copy = deepcopy(x)
    for arr in x_copy:
        feature = []
        for anchor in anchors:
            arr_anchor = anchor_mapping(arr, anchor)
            feature.append(merge_sort(arr_anchor))
        features.append(feature)
    features = np.array(features)
    features = np.mean(features, axis=1)
    normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2)
    return torch.tensor(features/normalizer)


def anchor_mapping(x, anchor):
    anchor_dict = {anchor[i]: i for i in range(len(anchor))}
    return [anchor_dict[int(i)] for i in x]


def evaluate_tsp(x, benchmark_index, dim):
    if x.dim() == 2:
        x = x.squeeze(0)
    x = x.numpy()
    A = np.asarray(scipy.io.loadmat('pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['A'])
    B = np.asarray(scipy.io.loadmat('pcb_dim_'+str(dim) + '_'+str(benchmark_index+1)+'.mat')['B'])
    E = np.eye(dim)

    permutation = np.array([np.arange(dim), x])

    P = np.zeros([dim, dim]) #initialize the permutation matrix

    for i in range(0,dim):
        P[:, i] = E[:, permutation[1][i]]

    result = (np.trace(P.dot(B).dot(P.T).dot(A.T)))
    print(f"Objective value: {result/10000}")
    return result/10000


def initialize_model(train_x, train_obj, covar_module=None, state_dict=None):
    # define models for objective and constraint
    if covar_module is not None:
        model = SingleTaskGP(train_x, train_obj, covar_module=covar_module)
    else:
        model = SingleTaskGP(train_x, train_obj)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()#noise_constraint=gpytorch.constraints.Positive())
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def EI_local_search(AF, x, anchors):
    feature = featurize(x.unsqueeze(0), anchors).unsqueeze(0).detach()
    best_val = AF(feature)
    best_point = x.numpy()
    for num_steps in range(100):
        # print(f"best AF value : {best_val} at best_point = {best_point}")
        all_vals = []
        all_points = []
        for i in range(len(best_point)):
            for j in range(i+1, len(best_point)):
                x_new = best_point.copy()
                x_new[i], x_new[j] = x_new[j], x_new[i]
                all_vals.append(AF(featurize(torch.from_numpy(x_new).unsqueeze(0), anchors).unsqueeze(1)).detach())
                all_points.append(x_new)
        idx = np.argmax(all_vals)
        if all_vals[idx] > best_val:
            best_point = all_points[idx]
            best_val = all_vals[idx]
        else:
              break
    print(f"best AF value : {best_val.item()} at best_point = {best_point}")
    return torch.from_numpy(best_point), best_val


def bo_loop(dim, benchmark_index, kernel_type, M=20):
    n_init = 20
    n_evals = 200
    for nruns in range(20):
        torch.manual_seed(nruns)
        np.random.seed(nruns)
        print(f'Input dimension {dim}')
        train_x = torch.from_numpy(np.array([np.random.permutation(np.arange(dim)) for _ in range(n_init)]))
        outputs = []
        for i in range(n_init):
            outputs.append(evaluate_tsp(train_x[i], benchmark_index, dim))
        train_y = -1*torch.tensor(outputs)
        anchors = [np.random.permutation(np.arange(dim)) for _ in range(M)]  # Random anchor initialization

        for num_iters in range(n_init, n_evals):

            inputs = featurize(train_x, anchors)
            if kernel_type == 'merge':
                covar_module = MergeKernel()
            train_y = (train_y - torch.mean(train_y))/(torch.std(train_y))
            mll_bt, model_bt = initialize_model(inputs, train_y.unsqueeze(1), covar_module)
            model_bt.likelihood.noise_covar.noise = torch.tensor(0.0001)
            mll_bt.model.likelihood.noise_covar.raw_noise.requires_grad = False
            fit_gpytorch_model(mll_bt)
            # print(train_y.dtype)
            print(f'\n -- NLL: {mll_bt(model_bt(inputs), train_y)}')
            EI = ExpectedImprovement(model_bt, best_f = train_y.max().item())
            # Multiple random restarts
            best_point, ls_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))), anchors)
            for _ in range(10):
                new_point, new_val = EI_local_search(EI, torch.from_numpy(np.random.permutation(np.arange(dim))), anchors)
                if new_val > ls_val:
                    best_point = new_point
                    ls_val = new_val
            print(f"Best Local search value: {ls_val}")
            if not torch.all(best_point.unsqueeze(0) == train_x, axis=1).any():
                best_next_input = best_point.unsqueeze(0)
            else:
                print(f"Generating randomly !!!!!!!!!!!")
                best_next_input = torch.from_numpy(np.random.permutation(np.arange(dim))).unsqueeze(0)
            # print(best_next_input)
            next_val = evaluate_tsp(best_next_input, benchmark_index, dim)
            train_x = torch.cat([train_x, best_next_input])
            outputs.append(next_val)
            train_y = -1*torch.tensor(outputs)
            # train_y = torch.cat([train_y, torch.tensor([next_val])])
            print(f"\n\n Iteration {num_iters} with value: {outputs[-1]}")
            print(f"Best value found till now: {np.min(outputs)}")
            torch.save({'inputs_selected':train_x, 'outputs':outputs, 'train_y':train_y}, 'tsp_botorch_'+kernel_type+'_EI_dim_'+str(dim)+'benchmark_index_group_average_'+str(benchmark_index)+'_nrun_'+str(nruns)+'.pkl')


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Bayesian optimization over permutations (QAP)')
    parser_.add_argument('--dim', dest='dim', type=int, default=10)
    parser_.add_argument('--benchmark_index', dest='benchmark_index', type=int, default=0)
    parser_.add_argument('--kernel_type', dest='kernel_type', type=str, default='merge')
    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    bo_loop(kwag_['dim'], kwag_['benchmark_index'], kwag_['kernel_type'])

