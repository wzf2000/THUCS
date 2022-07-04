#include "lib.h"
#include <string.h>
#include <time.h>

using namespace std;

int main() {
    bind_cpu();
    constexpr unsigned int size = 1024 * 64;
    constexpr unsigned int len = size / sizeof(int);
    constexpr unsigned int length_mod = len - 1;
    unsigned int base = 4;
    constexpr unsigned int steps = 1024 * 1024 * 128;
    clock_t begin, end;
    register int *a = new int[len];
    for (int _ = 0; _ < 5; ++_) {
        for (int i = 0; i < len; ++i)
            a[i] = 0;
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
    return 0;
}