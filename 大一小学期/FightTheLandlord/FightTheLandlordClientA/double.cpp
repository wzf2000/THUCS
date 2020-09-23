#include "double.h"

Double::Double(const Card *f, const Card *s) : Play(DOUBLE, 0)
{
    cards.push_back(const_cast<Card*>(f));
    cards.push_back(const_cast<Card*>(s));

    if (cards[0] < cards[1])
        std::swap(cards[0], cards[1]);
    num = getNum(cards[0]);
}

Double::Double(const QVector<Card*> &v) : Play(DOUBLE, 0, v)
{
    if (cards[0] < cards[1])
        std::swap(cards[0], cards[1]);
    num = getNum(cards[0]);
}

bool Double::operator<(const Double &rhs) const
{
    return num < rhs.num;
}

bool Double::operator>(const Double &rhs) const
{
    return num > rhs.num;
}

bool Double::isDouble(QVector<Card*> v)
{
    if (v.size() != 2)
        return false;
    Card *f = v[0], *s = v[1];
    if (f->isJoker || s->isJoker)
    {
        if (f->isJoker && s->isJoker)
            return f->type == s->type;
        return false;
    }
    return f->num == s->num;
}

bool Double::isLegal(QVector<Card *>v)
{
    if (!isDouble(v))
        return false;
    return *this < Double(v[0], v[1]);
}
