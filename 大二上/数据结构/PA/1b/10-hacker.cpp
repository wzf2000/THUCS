#include <iostream>

using namespace std;

int main()
{
    for (int i = 0; i < 2050; i++)
        putchar('A' + (i / 2) % 2);
    puts("");
    puts("1");
    puts("1025 A");
    return 0;
}
