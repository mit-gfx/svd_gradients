import numpy as np

###############################################################################
# Colorful prints.
###############################################################################
def print_error(*message):
    print('\033[91m', 'ERROR ', *message, '\033[0m')

def print_ok(*message):
    print('\033[92m', *message, '\033[0m')

def print_warning(*message):
    print('\033[93m', *message, '\033[0m')

def print_info(*message):
    print('\033[96m', *message, '\033[0m')

# Input:
# - x: any numpy array.
# Output:
# - The same array cast to double.
def ndarray(x):
    return np.asarray(np.copy(x), dtype=np.float64)

# Input:
# - n: dimension (2 or 3);
# Output:
# - R: a n x n matrix. |R| = 1. R * R.T = I.
def generate_random_rotation(n):
    assert n == 2 or n == 3
    U, _, _ = np.linalg.svd(np.random.normal(size=(n, n)))
    if np.linalg.det(U) < 0:
        U[0, :] = -U[0, :]
    return U

# Input:
# - sig: a n-D vector (n = 2 or 3) in desending order.
# Output:
# - A: a n x n matrix. |A| > 0 and _, sig, _ = np.linalg.svd(A).
def generate_random_matrix_from_singular_values(sig):
    sig = ndarray(sig).ravel()
    assert np.min(sig[:-1] - sig[1:]) >= 0
    n = sig.size
    U = generate_random_rotation(n)
    V = generate_random_rotation(n)
    return U @ np.diag(sig) @ V.T

# Input:
# - f: a function with the following signature: f_val, df = f(x, dx)
# - x0: the point where you would like to check gradients at.
# - eps: used to perturb x0.
# - atol, rtol: threshold used by np.allclose.
# Output:
# - Assertion failure if gradient check fails.
def check_grad(f, x0, eps, atol, rtol):
    x0 = ndarray(x0)
    dx = np.random.uniform(low=-eps, high=eps, size=x0.shape)
    f0, df = f(x0, dx)
    f1, _ = f(x0 + dx, dx)
    assert np.allclose(f1 - f0, df, atol=atol, rtol=rtol), print(f1 - f0, df)

# Input:
# - A: a n x n matrix where n = 2 or 3;
# - dA: a n x n matrix. Perturbation on A.
# Output:
# - sig: a n-D vector of singular values.
# - dsig: a n-D vector. Perturbation on sig.
def compute_sig(A, dA):
    A = ndarray(A)
    dA = ndarray(dA)
    assert A.shape == (2, 2) or A.shape == (3, 3)
    assert A.shape == dA.shape

    U, sig, Vt = np.linalg.svd(A)
    assert np.min(sig) > 0
    # Compute dsig.
    # eps controls when two singular values are considered to be the same.
    eps = 1e-10
    if np.min(sig[:-1] - sig[1:]) > eps:
        # General case: all sig are unique.
        dsig = np.diag(U.T @ dA @ Vt.T)
    elif np.max(sig) - np.min(sig) <= eps:
        # Special case: all sig are identical.
        # U * sig * Vt = A.
        # (At * A) * V = V * sig.
        # (A * At) * U = U * sig.
        # If we perturb A by dA:
        # (A + dA) * (At + dAt) = sig[0]^2 * I + dA * At + A * dAt.
        # Consider the eigenvectors of dA * At + A * dAt only:
        _, U = np.linalg.eigh(dA @ A.T + A @ dA.T)
        # w returned by eigh is in ascending order so U needs to be flipped.
        U = np.fliplr(U)
        Vt = np.diag(1.0 / sig) @ U.T @ A
        dsig = np.diag(U.T @ dA @ Vt.T)
    else:
        # Special case: two of the three sig are the same. There are two cases:
        # - sig[0] == sig[1] > sig[2].
        # - sig[0] > sig[1] == sig[2].
        n = U.shape[0]
        assert n == 3
        dsig = np.zeros(3)
        # As an example, consider sig[0] == sig[1] > sig[2]:
        # In this case, U and V are not uniquely defined. In particular,
        # consider an arbitrary 2 x 2 rotation matrix R and the following
        # modification on U and V (ui and vi are the i-th column of U and V):
        # U' = U; V' = V
        # [u'0, u'1] = [u0, u1] @ R
        # [v'0, v'1] = [v0, v1] @ R
        # It can be easily verified that U', sig, V' is still a valid SVD of A.

        # A = si * ui * vi.T
        #   = s0 * [u0, u1] @ [v0, v1].T + s2 * u2 @ v2.T
        # At = s0 * [v0, v1] @ [u0, u1].T + s2 * v2 @ u2.T
        # A * At = s0^2 * (I - u2 @ u2.T) + s2^2 * u2 @ u2.T
        #        = s0^2 * I + (s2^2 - s0^2) * u2 @ u2.T.
        if sig[0] - sig[1] <= eps:
            assert sig[1] - sig[2] > eps
            unique_idx = 2
        else:
            assert sig[1] - sig[2] <= eps
            unique_idx = 0
        u_unique = U[:, unique_idx]
        sig_unique = sig[unique_idx]
        sig_common = (np.sum(sig) - sig_unique) / 2
        _, U = np.linalg.eigh((sig_unique ** 2 - sig_common ** 2) * np.outer(u_unique, u_unique)
            + dA @ A.T + A @ dA.T)
        U = np.fliplr(U)
        Vt = np.diag(1.0 / sig) @ U.T @ A
        dsig = np.diag(U.T @ dA @ Vt.T)
    return sig, dsig