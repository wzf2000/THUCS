from re import M
from utils.utils import *

text = 'ADISPLAYEDEQUATION'
cipher = 'DSRMSIOPLXLJBZULLM'

input = [ch2num(ch) for ch in text]
output = [ch2num(ch) for ch in cipher]

m = 3

n = m ** 2 + m

A = [[0] * n for _ in  range(len(input))]
y = output
for i in range(len(input)):
    offset = i % m
    for j in range(m):
        A[i - offset + j][offset * m + j] = input[i]
    A[i][i % m + m ** 2] = 1

def inv(x):
    assert(x % 2 != 0)
    for i in range(N):
        if i * x % N == 1:
            return i

def div(x, y):
    assert(y % 2 == 1 or x % 2 == 0)
    if y % 2 == 0:
        return (x // 2 * inv(y // 2)) % N
    else:
        return (x * inv(y)) % N

def gauss(A, y, m, n, K):
    for i in range(n):
        max = i
        for j in range(i, m):
            if A[j][i] % 2 == 1 or A[j][i] > 0 and A[max][i] == 0:
                max = j
        tmp = A[max]
        A[max] = A[i]
        A[i] = tmp
        tmp = y[max]
        y[max] = y[i]
        y[i] = tmp
        assert(A[i][i] > 0)
        for j in range(i + 1, m):
            delta = div(A[j][i], A[i][i])
            for k in range(i, n):
                A[j][k] = (A[j][k] - delta * A[i][k] % N + N) % N
            y[j] = (y[j] - delta * y[i] % N + N) % N

    for i in range(n - 1, -1, -1):
        y[i] = div(y[i], A[i][i])
        A[i][i] = 1
        for j in range(i):
            y[j] = (y[j] - A[j][i] * y[i] % N + N) % N
            A[j][i] = 0

    L = y[:K * K]
    L = [L[i : i + K] for i in range(0, K * K, K)]
    b = y[K * K: n]
    return L, b

L, b = gauss(A, y, len(input), n, m)
print(L, b)
import numpy as np

output_text = ''
for i in range(len(input) // m):
    in_num = input[i * m : (i + 1) * m]
    out_num = (np.matmul(np.array(in_num), np.array(L)) + np.array(b)).tolist()
    for i in range(len(out_num)):
        out_num[i] %= N
        output_text += num2ch(out_num[i])
assert(output_text == cipher)
print(output_text)