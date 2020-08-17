import numpy as np
from svd_basic import compute_sig, generate_random_matrix_from_singular_values
from svd_basic import print_info, print_error, print_warning
from svd_basic import check_grad

def check_dsig(A, message):
    eps = 1e-4
    atol = 1e-6
    rtol = 1e-3
    try:
        check_grad(compute_sig, A, eps, atol, rtol)
        print_info('{} succeeded'.format(message))
    except AssertionError:
        print_error('{} failed'.format(message))

# Easy case: all singular values are unique.
n = 3
sig = np.random.uniform(low=0.1, high=1.0, size=n)
sig = -np.sort(-sig)    # Descending order.
A = generate_random_matrix_from_singular_values(sig)
check_dsig(A, 'unique singular values')

# A slightly more challenging case: all singular values are identical.
check_dsig(np.eye(n), 'identity matrix')
check_dsig(generate_random_matrix_from_singular_values([2, 2, 2]),
    'identical singular values')

# The most challenging case: two of the three singular values are identical.
check_dsig(generate_random_matrix_from_singular_values([2, 2, 1]),
    'singular values [2, 2, 1]')
check_dsig(generate_random_matrix_from_singular_values([2, 1, 1]),
    'singular values [2, 1, 1]')