#include <iostream>
#include "rand.h"

using namespace std;

const int NODES = 100000;
int nodes = NODES;
const int BLOCK_SIZE = 1;
int block_size;

int main(int argv, char *argc[])
{
    if (argv == 1)
        block_size = BLOCK_SIZE;
    else
        block_size = atoi(argc[1]);
    init_rand();
    int n = MAXN;
    if (argv >= 3)
        n = min(n, atoi(argc[2]));
    int qrys = n - nodes;
    int *a = new int[nodes];
    cout << n << '\n';
    for (int i = 0; i < nodes; ++i)
        a[i] = get_rand(MAXX / nodes * i, MAXX / nodes * (i + 1));
    for (int i = 0; i < nodes; ++i)
        cout << "A " << a[i] << '\n';
    int id = get_rand(nodes / 4, nodes / 4 * 3);
    for (int i = 0; i < qrys; ++i)
    {
        int x = a[id] + get_rand(0, block_size);
        cout << "C " << x << '\n';
    }
    return 0;
}
