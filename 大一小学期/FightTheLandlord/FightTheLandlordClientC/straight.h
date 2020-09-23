#ifndef STRAIGHT_H
#define STRAIGHT_H

#include "play.h"

template <unsigned n>
class Straight : public Play
{
public:
    Straight(const QVector<Card*>&);
    bool operator<(const Straight<n>&) const;
    bool operator>(const Straight<n>&) const;
    static bool isStraight(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

template <unsigned n>
Straight<n>::Straight(const QVector<Card*> &v) : Play(STRAIGHT, n, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });
    num = getNum(cards[0]);
}

template <unsigned n>
bool Straight<n>::operator<(const Straight &rhs) const
{
    return num < rhs.num;
}

template <unsigned n>
bool Straight<n>::operator>(const Straight &rhs) const
{
    return num > rhs.num;
}

template <unsigned n>
bool Straight<n>::isStraight(QVector<Card*> v)
{
    if (v.size() < 5 || v.size() != n)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v.back()->isJoker || v.back()->num == 2)
        return false;
    int f = v.first()->num;
    for (int i = 0; i < v.size(); i++)
    {
        int now = v[i]->num;
        if (now == 1) now += 13;
        if (now != f + i)
            return false;
    }
    return true;
}

template <unsigned n>
bool Straight<n>::isLegal(QVector<Card *>v)
{
    if (!isStraight(v))
        return false;
    return *this < Straight<n>(v);
}

#endif // STRAIGHT_H
