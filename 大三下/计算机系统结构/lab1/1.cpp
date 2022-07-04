#include "lib.h"
#include <string.h>
#include <time.h>

using namespace std;

int main() {
    bind_cpu();
    srand(time(0));
    unsigned int base = 1 * 1024;
    register unsigned int steps = 1024 * 1024 * 64;
    clock_t begin, end;
    for (int _ = 0; _ < 12; ++_) {
        cout << "test size = " << (base / 1024) << " KB, ";
        unsigned int len = base / sizeof(int);
        cout << "array length = " << len << ", ";
        register int *a = new int[len];
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
    return 0;
}