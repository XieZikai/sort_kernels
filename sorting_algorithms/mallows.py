import numpy as np


def featurize_mallows(x):
    """
    Featurize the permutation vector into continuous space. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    x_repeat = np.repeat(x, x.shape[-1], axis=1).reshape(x.shape[0], x.shape[1], -1)
    x_feature = np.transpose(x_repeat, (0, 2, 1)) - x_repeat
    x_feature = np.sign([i[np.triu_indices(x.shape[-1], k=1)] for i in x_feature])
    normalizer = np.sqrt(x.shape[1] * (x.shape[1] - 1) / 2)
    return x_feature / normalizer


def restore_featurize_mallows(x_feature, permutation=None):
    assert len(x_feature.shape) == 2 or len(x_feature.shape) == 1, "Only reverse featurize 1 or 2 dimension permutation vector"
    expand = False
    if len(x_feature.shape) == 1:
        x_feature = x_feature.reshape(1, -1)
        expand = True
    # x_feature = np.sign(x_feature)
    permutation_length = int(np.sqrt(x_feature.shape[-1] * 2)) + 1
    x = []
    for i in range(x_feature.shape[0]):
        x_i = np.zeros((permutation_length, permutation_length))
        row_idx = 0
        col_idx = row_idx + 1
        for num in x_feature[i]:
            x_i[row_idx, col_idx] = num
            if col_idx == permutation_length - 1:
                row_idx += 1
                col_idx = row_idx + 1
                continue
            col_idx += 1
        x_permutation_i = []
        for j in range(permutation_length):
            permutation_i = np.sum(x_i[j]) - np.sum(x_i[:, j])
            x_permutation_i.append(permutation_i)
        x_permutation_i = np.argsort(x_permutation_i)
        temp = [0 for i in range(permutation_length)]
        for j in range(permutation_length):
            temp[x_permutation_i[j]] = j
        x.append(temp)
    x = np.array(x)
    if expand:
        x = x[0]
    return x
