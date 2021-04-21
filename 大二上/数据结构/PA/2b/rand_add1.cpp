#include <iostream>
#include "rand.h"

using namespace std;

const int BLOCKS = 1;
int blocks;

int main(int argv, char *argc[])
{
    if (argv == 1)
        blocks = BLOCKS;
    else
        blocks = atoi(argc[1]);
    init_rand();
    int n = MAXN;
    if (argv >= 3)
        n = min(n, atoi(argc[2]));
    int *a = new int[n];
    cout << n << '\n';
    for (int i = 0; i < n; ++i)
        a[i] = get_rand(MAXX / n * i, MAXX / n * (i + 1));
    int block_size = n / blocks;
    for (int i = 0; i < blocks; ++i)
    {
        int l = i * block_size, r = (i + 1) * block_size;
        if (i == blocks - 1) r = n;
        my_shuffle(a + l, a + r);
    }
    for (int i = 0; i < n; ++i)
        cout << "A " << a[i] << '\n';
    return 0;
}
