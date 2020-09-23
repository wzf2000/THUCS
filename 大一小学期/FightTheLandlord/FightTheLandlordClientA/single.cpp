#include "single.h"

Single::Single(const Card *c) : Play(SINGLE, 0)
{
    cards.push_back(const_cast<Card*>(c));
    num = getNum(cards[0]);
}

Single::Single(const QVector<Card*> &v) : Play(SINGLE, 0, v)
{
    num = getNum(cards[0]);
}

bool Single::operator<(const Single &rhs) const
{
    return num < rhs.num;
}

bool Single::operator>(const Single &rhs) const
{
    return num > rhs.num;
}

bool Single::isSingle(QVector<Card *> v)
{
    return v.size() == 1;
}

bool Single::isLegal(QVector<Card *>v)
{
    if (!isSingle(v))
        return false;
    return *this < Single(v[0]);
}
