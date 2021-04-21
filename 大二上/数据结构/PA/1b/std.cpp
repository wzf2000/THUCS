#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace std;

char a[4000000];
int m, len;

void error(const char* msg) {
    printf("[Error] %s\n", msg);
    exit(1);
}

void check_initial_sequence() {
    if (len < 0 || 500000 < len) {
        error("Invalid initial length.");
    }

    char last_color = '\0';
    int last_counter = 0;
    for (int i = 0; i < len; ++i) {
        if (a[i] < 'A' || 'Z' < a[i]) {
            printf("%d %c\n", i, a[i]);
            error("Invalid color in initial sequence.");
        }
        if (a[i] == last_color) {
            ++last_counter;
            if (last_counter == 3) {
                error("Erasure should not happen in initial sequence.");
            }
        }
        else {
            last_color = a[i];
            last_counter = 1;
        }
    }
}

void play(int t) {
    int l = t - 1, r = t + 1;
    while (l >= 0 && a[l] == a[t]) --l;
    while (r < len && a[r] == a[t]) ++r;
    if (r - l > 3) {
        memmove(a + l + 1, a + r, len - r + 1);
        len -= (r - l - 1);
        if (l >= 0) play(l);
    }
}

int main() {
    fgets(a, 4000000, stdin);
    len = strlen(a);
    --len;
    a[len] = '\0';
    check_initial_sequence();

    if (scanf("%d", &m) != 1) {
        error("Read #operations error.");
    }
    if (!(0 <= m && m <= 500000)) {
        error("Invalid #operations.");
    }
    while (m--) {
        char col;
        int pos;
        if (scanf("%d %c", &pos, &col) != 2) {
            error("Operation format error.");
        }
        if (pos > len) {
            error("Invalid rank to insert.");
        }
        if (col < 'A' || 'Z' < col) {
            error("Invalid color to insert.");
        }
        memmove(a + pos + 1, a + pos, len - pos + 1);
        a[pos] = col;
        ++len;
        play(pos);
    }
    puts(a);
    return 0;
}
