#include <iostream>
#include <string>

using namespace std;

string a;

void play(int rank) {
    int lnum = 0, rnum = 0;

    for (int i = rank - 1; i >= 0; i--) {
        if (a[i] == a[rank]) lnum++;
        else break;
    }

    for (int i = rank + 1; i < a.size(); i++) {
        if (a[i] == a[rank]) rnum++;
        else break;
    }

    if (lnum + rnum + 1 > 2) {
        a.erase(rank - lnum, lnum + rnum + 1);
        play(rank - lnum - 1);
    }
}

int main() {
    cin >> a;
    int m = 0;
    cin >> m;

    int rank; char color;
    for (int i = 0; i < m; ++i) {
        cin >> rank >> color;
        a.insert(a.cbegin() + min(rank, (int)a.size()), color);
        play(rank);
    }

    cout << a << endl;

    return 0;
}
