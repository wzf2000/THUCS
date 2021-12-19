#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <omp.h>
#include <chrono>

void pow_a(int *a, int *b, int n, int m) {
    // TODO: 使用 omp parallel for 并行这个循环
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}

int main(int argc, char** argv) {
    // 运行参数检查
    if (argc != 4) {
        printf("Usage: ./openmp_pow n m seed\n");
        exit(1);
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int seed = atoi(argv[3]);
    int thread_count;

    // 获取运行的默认线程数
    #pragma omp parallel
    {
        #pragma omp master
        thread_count = omp_get_num_threads();
    }

    // 打印运行参数
    printf("openmp_pow: n = %d, m = %d, thread_count = %d\n", n, m, thread_count);
    fflush(stdout);

    // 设置随机种子
    srand(seed);

    // 分配内存
    int *a = new int[n];
    int *b = new int[n];
    
    // 初始化数组 a, b, c
    for (int i = 0; i < n; i++)
        a[i] = rand() % 1024;

    // 开始计时
    auto start = std::chrono::system_clock::now();

    // 计算 c = a + b
    pow_a(a, b, n, m);

    // 结束计时
    auto end = std::chrono::system_clock::now();

    // 检查结果
    int to_check = std::min(n, 10);
    for (int c = 0; c < to_check; c++) {
        int i = rand() % n;
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        if (b[i] != x) {
            printf("Wrong answer at position %d: %d != %d\n", i, b[i], x);
            exit(1);
        }
    }

    printf("Congratulations!\n");
    printf("Time Cost: %d us\n\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

    // 回收内存
    delete[] a;
    delete[] b;

    return 0;
}