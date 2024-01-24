'''
Description: 文件描述
Author: Cyletix
Date: 2023-04-02 23:38:20
LastEditTime: 2023-04-02 23:41:58
FilePath: \AutoMakeosuFile\svd.py
'''
import numpy as np

def svd_decomp(m, n, m1, n1):
    # Create a random matrix of size m x n
    A = np.random.rand(m, n)
    # Perform SVD decomposition
    U, s, V = np.linalg.svd(A)
    # Construct a diagonal matrix with singular values
    S = np.zeros((m, n))
    S[:n, :n] = np.diag(s)
    # Construct the output matrix
    B = U[:, :m1] @ S[:m1, :n1] @ V[:n1, :]
    return B[:m1,:n1]



# Example usage
if __name__=='__main__':
    B = svd_decomp(5, 4, 3, 2)
    print(B)