import numpy as np


def merge_sort(arr):
    # print(f'Handling array: {arr}')
    if len(arr) == 1:
        return []
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

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
                feature.append(0)
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
            feature.append(0)

        # print('Return: ', feature)
        return feature[:-1]


def featurize_merge(x):
    """
    Featurize the permutation vector into continuous space using merge kernel. Only available to permutation vector.
    """
    assert len(x.shape) == 2, "Only featurize 2 dimension permutation vector"
    feature = []
    for arr in x:
        feature.append(merge_sort(arr))
    return feature


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
