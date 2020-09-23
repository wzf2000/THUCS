#ifndef DOUBLESTRAIGHT_H
#define DOUBLESTRAIGHT_H

#include "play.h"

template <unsigned n>
class DoubleStraight : public Play
{
public:
    DoubleStraight(const QVector<Card*>&);
    bool operator<(const DoubleStraight<n>&) const;
    bool operator>(const DoubleStraight<n>&) const;
    static bool isDoubleStraight(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

template <unsigned n>
DoubleStraight<n>::DoubleStraight(const QVector<Card*> &v) : Play(DOUBLESTRAIGHT, n, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });
    num = getNum(cards[0]);
}

template <unsigned n>
bool DoubleStraight<n>::operator<(const DoubleStraight &rhs) const
{
    return num < rhs.num;
}

template <unsigned n>
bool DoubleStraight<n>::operator>(const DoubleStraight &rhs) const
{
    return num > rhs.num;
}

template <unsigned n>
bool DoubleStraight<n>::isDoubleStraight(QVector<Card*> v)
{
    if (v.size() < 6 || v.size() != n * 2)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v.back()->isJoker || v.back()->num == 2)
        return false;
    int f = v.first()->num;
    for (int i = 0; i < v.size(); i++)
    {
        int now = v[i]->num;
        if (now == 1) now += 13;
        if (now != f + i / 2)
            return false;
    }
    return true;
}

template <unsigned n>
bool DoubleStraight<n>::isLegal(QVector<Card*> v)
{
    if (!isDoubleStraight(v))
        return false;
    return *this < DoubleStraight<n>(v);
}

#endif // DOUBLESTRAIGHT_H
