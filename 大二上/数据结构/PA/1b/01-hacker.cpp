#include <iostream>

using namespace std;

int main()
{
    for (int i = 0; i < 500000; ++i)
        putchar(i % 26 + 'A');
    puts("");
    puts("500000");
    for (int i = 0; i < 500000; ++i)
        printf("260000 "), putchar(i % 26 + 'A'), puts("");
    return 0;
}
