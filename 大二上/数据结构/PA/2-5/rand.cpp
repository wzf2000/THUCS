#include <iostream>
#include <cstdlib>
using namespace std;
int main()
{
    srand(int(time(0)));
    int n = 200000;
    int m = 160000;
    cout << n << '\n';
    for (int i = 0; i < n; i++)
        cout << 0 << ' ' << rand() << ' ' << rand() << '\n';
    while (m--)
    {
        int y1 = rand(), y2 = rand();
        if (y1 > y2) swap(y1, y2);
        cout << 0 << ' ' << 0 << ' ' << y1 << ' ' << y2 << '\n';
    }
    return 0;
}