#include <iostream>
using namespace std;

char a[501001] = "";
int main()
{
    ios::sync_with_stdio(false);
    int n = 501000, k = 100;
    cout << n << " " << n << " " << k << '\n';
    for (int i = 0; i < n; i++)
        cout << (a[i] = (rand() % 10 + '0'));
    cout << '\n';
    for (int i = 0; i * 2 < k; i++)
        a[rand() % n] = rand() % 10 + '0';
    cout << a << "\n";
    return 0;
}
