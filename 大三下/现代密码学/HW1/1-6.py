from utils.utils import *

for k in range(N):
    flag = True
    for i in range(N):
        if add(add(i, k), k) != i:
            flag = False
            break

    if flag:
        print(k)
