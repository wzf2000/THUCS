#include <iostream>

using namespace std;

int main()
{
    for (int i = 0; i < (1 << 11) + 2; i++)
        putchar('A' + i % 26);
    puts("");
    puts("6");
    for (int i = 0; i < 3; i++)
        printf("%d A\n", (1 << 11) + 1);
    for (int i = 0; i < 3; i++)
        printf("%d %c\n", (1 << 11) + 1, 'B' + i);
    return 0;
}
