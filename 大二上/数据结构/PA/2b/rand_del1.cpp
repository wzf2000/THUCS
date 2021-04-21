#include <iostream>
#include "rand.h"

using namespace std;

const int NODES = 500000;
int nodes;

int main(int argv, char *argc[])
{
    if (argv == 1)
        nodes = NODES;
    else
        nodes = atoi(argc[1]);
    init_rand();
    int n = 2 * nodes;
    int *a = new int[nodes];
    cout << n << '\n';
    for (int i = 0; i < nodes; ++i)
        a[i] = get_rand(MAXX / nodes * i, MAXX / nodes * (i + 1));
    for (int i = 0; i < nodes; ++i)
        cout << "A " << a[i] << '\n';
    my_shuffle(a, a + nodes);
    for (int i = 0; i < nodes; ++i)
        cout << "B " << a[i] << '\n';
    return 0;
}
