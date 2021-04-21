#include <iostream>
#include <set>
#include "rand.h"

using namespace std;

set<int> st;

int main(int argv, char *argc[])
{
    init_rand();
    int n = MAXN;
    cout << n << '\n';
    int num = n / 10;
    int *a = new int [num];
    for (int i = 0; i < num; ++i)
        a[i] = get_rand(MAXN / 4 / num * i, MAXN / 4 / num * (i + 1)), st.insert(a[i]);
    my_shuffle(a, a + num);
    for (int i = 0; i < num; ++i)
        cout << "A " << a[i] << '\n';
    for (int i = num; i < n; ++i)
    {
        int op = get_rand(0, 2);
        if (op == 0)
        {
            int x = get_rand(0, MAXN / 4);
            auto it = st.lower_bound(x);
            if (*it == x)
                cout << "B " << x << '\n', st.erase(x);
            else
                cout << "A " << x << '\n', st.insert(x);
        }
        else
        {
            int x = get_rand(*st.begin(), MAXN / 4);
            cout << "C " << x << '\n';
        }
    }
    return 0;
}
