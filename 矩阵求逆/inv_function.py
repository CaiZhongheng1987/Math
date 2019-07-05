# encoding: utf-8
"""
@author: Cai Zhongheng
@contact: caiouyang4@gmail.com
@file: inv_function.py
@time: 2019/6/30 11:45
@desc: 使用QR分解来求解方阵的逆矩阵
"""
import numpy as np
import sys


def inv_qr(input_matrix):
    # 使用QR分解来求解逆矩阵，其中QR分解使用Givens旋转

    n = input_matrix.shape[0]
    R = input_matrix.copy()  # 深拷贝

    G = np.eye(n, dtype=complex)  # 初始化为单位矩阵
    Q = G.copy()

    # 从第0列开始，先从下往上，然后从左到右
    for col_idx in range(n-1):
        for row_idx in range(n-1, col_idx-1, -1):
            # 计算c和s
            if R[col_idx, col_idx] == 0:
                c = 0
                s = 1
            else:
                c = np.abs(R[col_idx, col_idx])/np.sqrt(R[col_idx, col_idx]*np.conj(R[col_idx, col_idx]) +
                                                        R[row_idx, col_idx]*np.conj(R[row_idx, col_idx]))
                s = c*R[row_idx, col_idx]/R[col_idx, col_idx]

            # 形成G矩阵
            G[col_idx, col_idx] = c
            G[row_idx, row_idx] = c
            G[col_idx, row_idx] = np.conj(s)
            G[row_idx, col_idx] = -s

            # 通过矩阵乘法将R和Q矩阵下三角的元素依次置0
            R.dot(G)
            Q.dot(G)
            # 将G矩阵恢复为单位矩阵，方便下次用
            G = np.eye(n, dtype=complex)

    # 计算上三角矩阵的逆矩阵
    inv_R = inv_upper_tri_matrix(R)
    return np.dot(inv_R, Q)


def inv_upper_tri_matrix(input_matrix):
    # 上三角矩阵的逆矩阵
    n = input_matrix.shape[0]

    if n is 1:
        return 1/input_matrix

    output_matrix = np.zeros((n, n), dtype=complex)
    for col_idx in range(n):
        # 先计算对角线上的元素
        if input_matrix[col_idx, col_idx] is 0:
            print('该矩阵不可逆！！！')
            sys.exit()
        else:
            output_matrix[col_idx, col_idx] = 1/input_matrix[col_idx, col_idx]

    # 从对角线开始往右斜上方推进
    for idx in range(1, n):
        row_idx = 0
        col_idx = idx
        while row_idx < n and col_idx < n:
            output_matrix[row_idx, col_idx] = -1*np.dot(input_matrix[row_idx, (row_idx+1):(col_idx+1)],
                                                        output_matrix[(row_idx+1):(col_idx+1), col_idx])
            output_matrix[row_idx, col_idx] /= input_matrix[row_idx, row_idx]
            row_idx += 1
            col_idx += 1

    return output_matrix


if __name__ == '__main__':
    np.random.seed(23)
    matrix_len = 4
    input_matrix = np.random.random((matrix_len, matrix_len)) + 1j * np.random.random((matrix_len, matrix_len))
    # input_matrix = np.triu(input_matrix)
    print(input_matrix)

    inv_matrix = inv_qr(input_matrix)
    # inv_matrix = inv_upper_tri_matrix(input_matrix)
    py_inv = np.linalg.inv(inv_matrix)
    print(np.dot(inv_matrix, py_inv))
