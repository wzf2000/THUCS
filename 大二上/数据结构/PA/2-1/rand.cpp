#include <iostream>
#include <cstdlib>

using namespace std;

int main()
{
    srand(int(time(0)));
    int n = 7, m = 5;
    cout << n << " " << m << "\n";
    cout << n - 1;
    for (int i = 2; i <= n; i++)
        cout << " " << i;
    cout << "\n";
    for (int i = 2; i <= n; i++)
        cout << 0 << "\n";
    for (int i = 0; i < m; i++)
    {
        int op = rand() % 3;
        cout << op << '\n';
        if (op)
        {
            int num = rand() % n;
            cout << num << ' ';
            for (int i = 0; i < num; i++)
                cout << i % 5 << ' ';
            cout << '\n';
        }
        else
        {
            int num = rand() % n;
            cout << num + 1 << ' ' << 0 << ' ';
            for (int i = 0; i < num; i++)
                cout << rand() % 5 << ' ';
            cout << '\n';
            num = rand() % n;
            cout << num << ' ';
            for (int i = 0; i < num; i++)
                cout << rand() % 5 << ' ';
            cout << '\n';
            cout << 0 << '\n';
        }
    }
    return 0;
}
