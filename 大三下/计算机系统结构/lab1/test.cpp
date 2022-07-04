#include <iostream>
#include <string.h>
#include <time.h>

using namespace std;

void bind_cpu() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(2, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask) < 0) {
        perror("sched_setaffinity");
    }
}

void test1() {
    unsigned int base = 1 * 1024;
    register unsigned int steps = 1024 * 1024 * 512;
    clock_t begin, end;
    for (int _ = 0; _ < 14; ++_) {
        cout << "test size = " << (base / 1024) << " KB, ";
        unsigned int len = base / sizeof(int);
        cout << "array length = " << len << ", ";
        register int *a = new int[len];
        for (register int i = 0; i < len; ++i) {
            a[i] = 0;
        }
        register unsigned int length_mod = len - 1;
        begin = clock();
        for (register int i = 0; i < steps; ++i) {
            a[(i * 16) & length_mod]++;
        }
        end = clock();
        cout << "end - begin = " << (double)(end - begin) / CLOCKS_PER_SEC * 1000 << " ms\n";
        delete [] a;
        base *= 2;
    }
}

void test2() {
    constexpr unsigned int size = 1024 * 64;
    constexpr unsigned int len = size / sizeof(int);
    constexpr unsigned int length_mod = len - 1;
    unsigned int base = 4;
    constexpr unsigned int steps = 1024 * 1024 * 128;
    clock_t begin, end;
    register int *a = new int[len];
    for (int _ = 0; _ < 6; ++_) {
        for (register int i = 0; i < len; ++i) {
            a[i] = 0;
        }
        cout << "test block size = " << base << " B, ";
        register unsigned int step = base / sizeof(int);
        cout << "step = " << step << ", ";
        begin = clock();
        for (register int i = 0; i < steps; ++i) {
            a[(i * step) & length_mod] = 1;
        }
        end = clock();
        cout << "end - begin = " << (double)(end - begin) / CLOCKS_PER_SEC * 1000 << " ms\n";
        base *= 2;
    }
    delete [] a;
}

void test3() {
    constexpr unsigned int size = 1024 * 64;
    constexpr unsigned int len = size / sizeof(int);
    constexpr unsigned int steps = 1024 * 1024 * 64;
    clock_t begin, end;
    register int *a = new int[len];
    for (register unsigned int base = 4; base <= 256; base *= 2) {
        cout << "test number of groups = " << base << ", ";
        register unsigned int block_size = len / base;
        register unsigned int step = block_size * 2;
        cout << "block size = " << block_size << ", ";
        register unsigned int pos = block_size + block_size / 2;
        begin = clock();
        for (register int i = 0; i < steps; ++i) {
            a[pos]++;
            pos += step;
            if (pos >= len)
                pos -= len;
        }
        end = clock();
        cout << "end - begin = " << (double)(end - begin) / CLOCKS_PER_SEC * 1000 << " ms\n";
    }
    delete [] a;
}

int main() {
    bind_cpu();
    cout << "\033[1m\033[31mTest cache size:\033[0m\n";
    test1();
    cout << "\033[1m\033[31mTest cache line size:\033[0m\n";
    test2();
    cout << "\033[1m\033[31mTest cache associativity:\033[0m\n";
    test3();
    return 0;
}