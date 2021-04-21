#include <iostream>
#include <string>

using namespace std;

string a;

void play(int rank) {
    int left = rank;
    int right = rank;
    char color = a[rank];

    while (left >= 0 && a[left] == color) --left;
    left += 1;
    while (right < a.size() && a[right] == color) ++right;

    int size = right - left;
    if (size >= 3) {
        a.erase(left, size);

        int next;
        if (left - 1 >= 0) next = left - 1;
        else if (left < a.size()) next = left;
        else return;

        play(next);
    }
}

int main() {
    getline(cin, a);
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
