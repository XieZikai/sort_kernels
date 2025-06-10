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
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound
from gpytorch.constraints import Interval, Positive
from gpytorch.priors import Prior
from gpytorch.kernels import Kernel
from copy import deepcopy
from botorch.optim import optimize_acqf


class MergeKernel(Kernel):
    has_lengthscale = True
    def forward(self, x1, x2, **params):
        x1_unsq = x1.unsqueeze(-2)
        x2_unsq = x2.unsqueeze(-3)
        diff = x1_unsq - x2_unsq
        diff_sq = diff.pow(2).sum(dim=-1)
        return torch.exp(-self.lengthscale * diff_sq)


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



def featurize(x, anchor):
    """
    Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    feature = []
    x_copy = deepcopy(x)
    for arr in x_copy:
        arr_anchor = anchor_mapping(arr, anchor)
        feature.append(merge_sort(arr_anchor))
    normalizer = np.sqrt(x.size(1)*(x.size(1) - 1)/2)
    return torch.tensor(feature/normalizer)


def anchor_mapping(x, anchor):
    # return x  # Anchor not used in this version.
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


def restore_merge_sort(arr, permutation):
    # print(f'Handling array: {arr} with permutation {permutation}')
    if len(permutation) == 2:
        if arr[0] == 0:
            return permutation
        else:
            return [permutation[1], permutation[0]]
    if len(permutation) == 3:
        if arr[1] + arr[2] == 0:
            right_permutation = [permutation[1], permutation[2]]
            left_permutation = [permutation[0]]
        elif arr[1] + arr[2] == 1:
            right_permutation = [permutation[0], permutation[2]]
            left_permutation = [permutation[1]]
        else:
            right_permutation = [permutation[0], permutation[1]]
            left_permutation = [permutation[2]]

        right_permutation = [right_permutation[0], right_permutation[1]] if arr[0] == 0 else [right_permutation[1],
                                                                                              right_permutation[0]]
        return left_permutation + right_permutation

    permutation_length = len(permutation)
    order = arr[-permutation_length + 1:]
    arr = arr[:-permutation_length + 1]

    left_permutation = []
    right_permutation = []

    # print('order: ', order)
    for index, i in enumerate(order):
        if i == 0:
            left_permutation.append(permutation[index])
        else:
            right_permutation.append(permutation[index])

        if len(left_permutation) == len(permutation) // 2:
            for j in range(index + 1, len(permutation)):
                right_permutation.append(permutation[j])
            break
        elif len(right_permutation) == int(np.ceil(len(permutation) / 2)):
            for j in range(index + 1, len(permutation)):
                left_permutation.append(permutation[j])
            break

    # print('left & right: ', left_permutation, right_permutation)
    if len(left_permutation) == len(right_permutation):
        mid = len(arr) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]
    else:
        difference = int(np.floor(np.log2(len(left_permutation))) + 1)
        # print(difference)
        mid = (len(arr) - difference) // 2
        left_arr = arr[:mid]
        right_arr = arr[mid:]

    left = restore_merge_sort(left_arr, left_permutation)
    right = restore_merge_sort(right_arr, right_permutation)
    return left + right


def restore_featurize_merge(x, permutation):
    feature = []
    for arr in x:
        arr_norm = []
        for i in arr:
            if i > 0:
                arr_norm.append(1)
            else:
                arr_norm.append(0)
        feature.append(restore_merge_sort(arr_norm, permutation))
    return feature


def EI_optimize(AF, x, anchor, num_restarts=50, raw_samples=100, n_iter=1000):
    feature = featurize(x.unsqueeze(0), anchor).unsqueeze(0).detach()
    feature_length = feature.shape[-1]
    permutation_length = x.shape[-1]

    bounds = torch.stack([-torch.ones(feature_length), torch.ones(feature_length)])
    candidates, acq_value = optimize_acqf(
        acq_function=AF,  # 已经构造好的 ExpectedImprovement 实例
        bounds=bounds,  # shape = (2, d)
        q=1,  # 同时找 1 个候选点
        num_restarts=num_restarts,  # 重启次数
        raw_samples=raw_samples,  # 每次重启时的随机初始点数量
        options={"maxiter": n_iter},
    )

    permutation = restore_featurize_merge(candidates, [i for i in range(permutation_length)])[0]
    print(f"best AF value : {permutation} at best_point = {acq_value.item()}")
    return torch.tensor(permutation), acq_value.item()


def EI_local_search(AF, x, anchor):
    feature = featurize(x.unsqueeze(0), anchor).unsqueeze(0).detach()
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
                all_vals.append(AF(featurize(torch.from_numpy(x_new).unsqueeze(0), anchor).unsqueeze(1)).detach())
                all_points.append(x_new)
        idx = np.argmax(all_vals)
        if all_vals[idx] > best_val:
            best_point = all_points[idx]
            best_val = all_vals[idx]
        else:
              break
    print(f"best AF value : {best_val.item()} at best_point = {best_point}")
    return torch.from_numpy(best_point), best_val


def bo_loop(dim, benchmark_index, kernel_type):
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

        anchor = train_x[train_y.argmax()].numpy()

        for num_iters in range(n_init, n_evals):
            inputs = featurize(train_x, anchor)
            if kernel_type == 'merge':
                covar_module = MergeKernel()
            train_y = (train_y - torch.mean(train_y))/(torch.std(train_y))
            mll_bt, model_bt = initialize_model(inputs, train_y.unsqueeze(1), covar_module)
            model_bt.likelihood.noise_covar.noise = torch.tensor(0.0001)
            mll_bt.model.likelihood.noise_covar.raw_noise.requires_grad = False
            fit_gpytorch_model(mll_bt)
            # print(train_y.dtype)
            print(f'\n -- NLL: {mll_bt(model_bt(inputs), train_y)}')
            EI = UpperConfidenceBound(model_bt, beta=2.576)
            # EI = ExpectedImprovement(model_bt, best_f = train_y.max().item())
            # Multiple random restarts
            best_point, ls_val = EI_optimize(EI, torch.from_numpy(np.random.permutation(np.arange(dim))), anchor)
            for _ in range(1):
                new_point, new_val = EI_optimize(EI, torch.from_numpy(np.random.permutation(np.arange(dim))), anchor)
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
            torch.save({'inputs_selected':train_x, 'outputs':outputs, 'train_y':train_y}, 'tsp_botorch_'+kernel_type+'_UCB_dim_'+str(dim)+'benchmark_index_ei_opt_'+str(benchmark_index)+'_nrun_'+str(nruns)+'.pkl')


if __name__ == '__main__':
    parser_ = argparse.ArgumentParser(
        description='Bayesian optimization over permutations (QAP)')
    parser_.add_argument('--dim', dest='dim', type=int, default=10)
    parser_.add_argument('--benchmark_index', dest='benchmark_index', type=int, default=0)
    parser_.add_argument('--kernel_type', dest='kernel_type', type=str, default='merge')
    args_ = parser_.parse_args()
    kwag_ = vars(args_)
    bo_loop(kwag_['dim'], kwag_['benchmark_index'], kwag_['kernel_type'])

