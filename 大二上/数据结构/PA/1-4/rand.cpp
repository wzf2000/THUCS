#include <iostream>

using namespace std;

int main()
{
    srand(time(0));
    ios::sync_with_stdio(false);
    int n = 1000000;
    cout << n << '\n';
    for (int i = 1; i <= n; i++)
        cout << rand() % 2000000 + 1 << (i == n - 1 ? '\n' : ' ');
    int now = 0;
    for (int i = 1; i <= n; i++)
    {
        cout << (i - now) << (i == n - 1 ? '\n' : ' ');
        now += rand() & 1;
    }
    int T = 100000;
    cout << T << '\n';
    while (T--)
    {
        unsigned int p, q;
        if (T <= 10)
            p = static_cast<unsigned int>(rand()) + static_cast<unsigned int>(rand()), q = static_cast<unsigned int>(rand()) + static_cast<unsigned int>(rand());
        else
            p = rand() % 2000000 + 1, q = rand() % 2000000 + 1;
        if (p > q) swap(p, q);
        if (p == q) q++;
        cout << p << " " << q << "\n";
    }
    return 0;
}
