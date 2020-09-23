#ifndef TRIPLE_H
#define TRIPLE_H

#include "play.h"

template <unsigned n>
class Triple : public Play
{
public:
    Triple(const QVector<Card*>&);
    bool operator<(const Triple<n>&) const;
    bool operator>(const Triple<n>&) const;
    static bool isTriple(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

template <unsigned n>
Triple<n>::Triple(const QVector<Card*> &v) : Play(TRIPLE, n, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });
    num = getNum(cards[0]);
}

template <unsigned n>
bool Triple<n>::operator<(const Triple &rhs) const
{
    return num < rhs.num;
}

template <unsigned n>
bool Triple<n>::operator>(const Triple &rhs) const
{
    return num > rhs.num;
}

template <unsigned n>
bool Triple<n>::isTriple(QVector<Card*> v)
{
    if (v.size() != n * 3)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v.back()->isJoker)
        return false;
    if (v.back()->num == 2)
    {
        if (v.size() != 3)
            return false;
        if (v[0]->num != 2 || v[1]->num != 2)
            return false;
        return true;
    }
    int f = v.first()->num;
    for (int i = 0; i < v.size(); i++)
    {
        int now = v[i]->num;
        if (now == 1) now += 13;
        if (now != f + i / 3)
            return false;
    }
    return true;
}

template <unsigned n>
bool Triple<n>::isLegal(QVector<Card *>v)
{
    if (!isTriple(v))
        return false;
    return *this < Triple<n>(v);
}

#endif // TRIPLE_H
