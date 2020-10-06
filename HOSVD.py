def hosvd(X):
    U = [None for _ in range(X.ndims())]
    dims = X.ndims()
    S = X
    for d in range(dims):
        # mode n分解
        C = base.unfold(X, d)
        # SVD分解
        U1, S1, V1 = np.linalg.svd(C)
        # 迭代求解核心張量
        S = base.tensor_times_mat(S, U1.T, d)
        U[d] = U1
    core = S
    # 回傳伴隨矩陣和核心張量
    return U, core
