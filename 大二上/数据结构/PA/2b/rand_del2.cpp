#include <iostream>
#include "rand.h"

using namespace std;

const int NODES = (1 << 19) * 3 - 1;
int nodes = NODES;
int *id, *a;
int cnt = 0;

void dfs(int cur)
{
    if (cur > nodes) return;
    dfs(cur << 1);
    id[cur - 1] = cnt++;
    dfs(cur << 1 | 1);
}


int main(int argv, char *argc[])
{
    init_rand();
    int n = nodes + 250000 * 2;
    a = new int[nodes];
    id = new int[nodes];
    cout << n << '\n';
    for (int i = 0; i < nodes; ++i)
        a[i] = get_rand(MAXX / nodes * i, MAXX / nodes * (i + 1));
    dfs(1);
    for (int i = 0; i < nodes; ++i)
        cout << "A " << a[id[i]] << '\n';
    for (int i = 0; i < 250000; ++i)
    {
        cout << "B " << a[nodes - 1] << '\n';
        cout << "A " << a[nodes - 1] << '\n';
    }
    return 0;
}
