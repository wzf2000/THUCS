#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <mpi.h>
#include <chrono>

void pow_a(int *a, int *b, int n, int m, int comm_sz /* 总进程数 */) {
    // TODO: 对这个进程拥有的数据计算 b[i] = a[i]^m
    for (int i = 0; i < n / comm_sz; ++i) {
        int x = 1;
        for (int j = 0; j < m; ++j)
            x *= a[i];
        b[i] = x;
    }
}

int main(int argc, char** argv) {
    // MPI 系统的初始化
    MPI_Init(nullptr, nullptr);
    // 获取总进程数
    int comm_sz;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    // 获取当前进程的进程号
    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    // 运行参数检查
    if (argc != 4) {
        printf("Usage: ./mpi_pow n m seed\n");
        // MPI 系统的终止
        MPI_Finalize();
        exit(1);
    }
    int n = atoi(argv[1]);
    int m = atoi(argv[2]);
    int seed = atoi(argv[3]);
    if (n % comm_sz != 0) {
        printf("Invalid parameter: n % comm_sz != 0\n");
        // MPI 系统的终止
        MPI_Finalize();
        exit(1);
    }

    // 打印运行参数
    if (my_rank == 0) {
        printf("mpi_pow: n = %d, m = %d, process_count = %d\n", n, m, comm_sz);
        fflush(stdout);
    }

    // 设置随机种子
    srand(seed);

    // 分配内存
    int *a = new int[n / comm_sz];
    int *b = new int[n / comm_sz];
    int *root_a, *root_b;

    // 进程 0 负责生成数组 a 的所有数据
    if (my_rank == 0) {
        root_a = new int[n];
        root_b = new int[n];

        for (int i = 0; i < n; i++)
            root_a[i] = rand() % 1024;
    }

    // 开始计时
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::system_clock::now();
    
    // 进程 i 获取进程 0 中位于 [my_rank * (n / comm_sz), (my_rank + 1) * (n / comm_sz)) 位置的数据
    MPI_Scatter(
        root_a, n / comm_sz, MPI_INT, // send_buf_p, send_count, send_type
        a, n / comm_sz, MPI_INT,      // recv_buf_p, recv_count, recv_type
        0, MPI_COMM_WORLD             // src_process, comm
    );
    
    // 计算 b[i] = a[i]^(m)
    pow_a(a, b, n, m, comm_sz);

    // 进程 0 收集各进程的运算结果
    MPI_Gather(
        b, n / comm_sz, MPI_INT,      // send_buf_p, send_count, send_type
        root_b, n / comm_sz, MPI_INT, // recv_buf_p, recv_count, recv_type
        0, MPI_COMM_WORLD             // src_process, comm
    );

    // 结束计时
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::system_clock::now();

    // 进程 0 检查结果并打印运行时间
    if (my_rank == 0) {
        int to_check = std::min(n, 10);
        for (int c = 0; c < to_check; c++) {
            int i = rand() % n;
            int x = 1;
            for (int j = 0; j < m; j++)
                x *= root_a[i];
            if (root_b[i] != x) {
                printf("Wrong answer at position %d: %d != %d\n", i, b[i], x);
                // MPI 系统的终止
                MPI_Finalize();
                exit(1);
            }
        }

        printf("Congratulations!\n");
        printf("Time Cost: %d us\n\n", std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

        // 回收内存
        delete[] root_a;
        delete[] root_b;
    }

    // 回收内存
    delete[] a;
    delete[] b;

    // MPI 系统的终止
    MPI_Finalize();

    return 0;
}