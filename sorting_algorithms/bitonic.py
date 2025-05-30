def bitonic_sort_iterative(arr):
    """维基百科伪代码改写的 Bitonic Sort（迭代版）只能对2^n长度的数组有用"""
    n = len(arr)
    arr = arr[:]
    trace = []
    trace_set = []
    embeddings = []

    k = 2
    while k <= n:
        j = k // 2
        while j > 0:
            for i in range(n):
                l = i ^ j
                if l > i:
                    if ((i & k) == 0 and arr[i] > arr[l]) or ((i & k) != 0 and arr[i] < arr[l]):
                        arr[i], arr[l] = arr[l], arr[i]
                        trace.append(arr[:])
                        trace_set.append([i, l])
                        embeddings.append(1)
                    else:
                        trace.append(arr[:])
                        trace_set.append([i, l])
                        embeddings.append(0)
            j //= 2
        k *= 2

    return arr, trace, trace_set, embeddings


def bitonic_any_length(arr):
    n = 0
    while 2 ** n < len(arr):
        n += 1
    padded_arr = arr + list(range(len(arr), 2 ** n))
    return padded_arr, 2 ** n - len(arr)

