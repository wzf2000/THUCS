#include "quartetplustwodouble.h"

QuartetPlusTwoDouble::QuartetPlusTwoDouble(const QVector<Card*> &v) : Play(QUARTEPLUSTWODOUBLE, 0, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (cards[4]->num == cards[5]->num && cards[5]->num == cards[6]->num && cards[6]->num == cards[7]->num && cards[0]->num == cards[1]->num && cards[2]->num == cards[3]->num)
    {
        num = getNum(cards[4]);
        std::swap(cards[0], cards[4]);
        std::swap(cards[1], cards[5]);
        std::swap(cards[2], cards[6]);
        std::swap(cards[3], cards[7]);
    }
    else
        if (cards[2]->num == cards[3]->num && cards[3]->num == cards[4]->num && cards[4]->num == cards[5]->num && cards[0]->num == cards[1]->num && cards[6]->num == cards[7]->num)
        {
            num = getNum(cards[2]);
            std::swap(cards[0], cards[4]);
            std::swap(cards[1], cards[5]);
        }
        else
            num = getNum(cards[0]);
}

bool QuartetPlusTwoDouble::operator<(const QuartetPlusTwoDouble &rhs) const
{
    return num < rhs.num;
}

bool QuartetPlusTwoDouble::operator>(const QuartetPlusTwoDouble &rhs) const
{
    return num > rhs.num;
}

bool QuartetPlusTwoDouble::isQuartetPlusTwoDouble(QVector<Card*> v)
{
    if (v.size() != 8)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v[0]->num == v[1]->num && v[1]->num == v[2]->num && v[2]->num == v[3]->num && v[4]->num == v[5]->num && v[6]->num == v[7]->num)
        return true;
    if (v[2]->num == v[3]->num && v[3]->num == v[4]->num && v[4]->num == v[5]->num && v[0]->num == v[1]->num && v[6]->num == v[7]->num)
        return true;
    if (v[4]->num == v[5]->num && v[5]->num == v[6]->num && v[6]->num == v[7]->num && v[0]->num == v[1]->num && v[2]->num == v[3]->num)
        return true;
    return false;
}

bool QuartetPlusTwoDouble::isLegal(QVector<Card *>v)
{
    if (!isQuartetPlusTwoDouble(v))
        return false;
    return *this < QuartetPlusTwoDouble(v);
}
