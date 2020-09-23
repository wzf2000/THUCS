#include "bomb.h"

Bomb::Bomb(const QVector<Card*> &v) : Play(BOMB, 0, v)
{
    num = getNum(cards[0]);
}

bool Bomb::operator<(const Bomb &rhs) const
{
    return num < rhs.num;
}

bool Bomb::operator>(const Bomb &rhs) const
{
    return num > rhs.num;
}

bool Bomb::operator<(const JokerBomb&) const
{
    return true;
}

bool Bomb::operator>(const JokerBomb&) const
{
    return false;
}

bool Bomb::isBomb(QVector<Card*> v)
{
    if (v.size() != 4)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });
    if (v.back()->isJoker)
        return false;
    int f = v.first()->num;
    for (int i = 0; i < v.size(); i++)
    {
        int now = v[i]->num;
        if (now != f)
            return false;
    }
    return true;
}

bool Bomb::isLegal(QVector<Card *>v)
{
    if (!isBomb(v))
        return false;
    return *this < Bomb(v);
}
