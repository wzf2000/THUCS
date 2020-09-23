#include "jokerbomb.h"

JokerBomb::JokerBomb(const Card *f, const Card *s) : Play(JOKERBOMB, 0)
{
    cards.push_back(const_cast<Card*>(f));
    cards.push_back(const_cast<Card*>(s));

    if (cards[0] < cards[1])
        std::swap(cards[0], cards[1]);
}

JokerBomb::JokerBomb(const QVector<Card*> &v) : Play(JOKERBOMB, 0, v)
{
    if (cards[0] < cards[1])
        std::swap(cards[0], cards[1]);
}

bool JokerBomb::operator<(const Bomb&) const
{
    return false;
}

bool JokerBomb::operator>(const Bomb&) const
{
    return true;
}

bool JokerBomb::isJokerBomb(QVector<Card*> v)
{
    if (v.size() != 2)
        return false;
    Card *f = v[0], *s = v[1];
    if (!f->isJoker || !s->isJoker)
        return false;
    return f->type != s->type;
}

bool JokerBomb::isLegal(QVector<Card*>)
{
    return false;
}
