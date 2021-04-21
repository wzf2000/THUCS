#include <iostream>

using namespace std;

int main()
{
    for (int i = 0; i < 333333; i++)
        putchar('A' + i % 2);
    putchar('A');
    for (int i = 0; i < 166666; i++)
        putchar('A' + ((i / 2 % 2) ^ 1));
    puts("");
    puts("500000");
    for (int i = 166666; i < 666665; i++)
        printf("%d %c\n", i + 333334, 'A' + ((i / 2 % 2) ^ 1));
    puts("333333 A");
    return 0;
}
