#include <iostream>

using namespace std;

int main()
{
    for (int i = 0; i < 999999; i++)
        cout << (i & 1 ? (rand() & 1 ? '+' : '-') : (rand() & 1 ? 'x' : '1'));
    cout << '\n';
    return 0;
}
