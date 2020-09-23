#include "quartetplustwosingle.h"

QuartetPlusTwoSingle::QuartetPlusTwoSingle(const QVector<Card*> &v) : Play(QUARTEPLUSTWOSINGLE, 0, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (cards[0]->num == cards[1]->num && cards[1]->num == cards[2]->num)
        num = getNum(cards[0]);
    else
        if (cards[1]->num == cards[2]->num && cards[2]->num == cards[3]->num)
        {
            num = getNum(cards[1]);
            std::swap(cards[0], cards[4]);
        }
        else
        {
            num = getNum(cards[2]);
            std::swap(cards[0], cards[4]);
            std::swap(cards[1], cards[5]);
        }
}

bool QuartetPlusTwoSingle::operator<(const QuartetPlusTwoSingle &rhs) const
{
    return num < rhs.num;
}

bool QuartetPlusTwoSingle::operator>(const QuartetPlusTwoSingle &rhs) const
{
    return num > rhs.num;
}

bool QuartetPlusTwoSingle::isQuartetPlusTwoSingle(QVector<Card*> v)
{
    if (v.size() != 6)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v[0]->num == v[1]->num && v[1]->num == v[2]->num && v[2]->num == v[3]->num)
        return true;
    if (v[1]->num == v[2]->num && v[2]->num == v[3]->num && v[3]->num == v[4]->num)
        return true;
    if (v[2]->num == v[3]->num && v[3]->num == v[4]->num && v[4]->num == v[5]->num)
        return true;
    return false;
}

bool QuartetPlusTwoSingle::isLegal(QVector<Card *>v)
{
    if (!isQuartetPlusTwoSingle(v))
        return false;
    return *this < QuartetPlusTwoSingle(v);
}
