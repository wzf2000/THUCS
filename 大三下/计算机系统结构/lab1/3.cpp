#include "lib.h"
#include <string.h>
#include <time.h>

using namespace std;

int main() {
    bind_cpu();
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
    return 0;
}
