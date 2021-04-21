#include <iostream>

using namespace std;

int main()
{
    srand((int)time(0));
    puts("10");
    for (int i = 0; i < 10;i++)
        printf("%d%c", rand() % 10 + 1, i == 9 ? '\n' : ' ');
    for (int i = 0; i < 10;i++)
        printf("%d%c", rand() % 10 + 1, i == 9 ? '\n' : ' ');
    puts("10");
    for (int i = 0; i < 10;i++)
        printf("%d %d\n", rand() % 10 + 1, rand() % 10 + 1);
    return 0;
}
