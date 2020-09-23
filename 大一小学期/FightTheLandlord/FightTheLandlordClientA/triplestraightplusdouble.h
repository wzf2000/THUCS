#ifndef TRIPLESTRAIGHTPLUSDOUBLE_H
#define TRIPLESTRAIGHTPLUSDOUBLE_H

#include "play.h"

template <unsigned n>
class TripleStraightPlusDouble : public Play
{
public:
    TripleStraightPlusDouble(const QVector<Card*>&);
    bool operator<(const TripleStraightPlusDouble<n>&) const;
    bool operator>(const TripleStraightPlusDouble<n>&) const;
    static bool isTripleStraightPlusDouble(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

template <unsigned n>
TripleStraightPlusDouble<n>::TripleStraightPlusDouble(const QVector<Card*> &v) : Play(TRIPLESTRAIGHTPLUSDOUBLE, n, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });

    int cnt[18] = {};
    for (auto card : cards)
        cnt[getNum(card)]++;
    for (int i = 15 - n + (n == 1); i >= 3; i--)
    {
        bool flag = true;
        for (int j = i; j < i + int(n); j++)
            if (cnt[j] != 3)
                flag = false;
        if (flag)
        {
            num = i;
            QVector<Card*> temp(cards);
            cards.clear();
            QVector<Card*>::iterator it = temp.begin();
            int pos = 0;
            for (int j = i; j < i + int(n); j++)
            {
                while (getNum(*(it + pos)) != j)
                    pos++;
                cards.push_back(*(it + pos));
                temp.erase(it + pos);
                cards.push_back(*(it + pos));
                temp.erase(it + pos);
                cards.push_back(*(it + pos));
                temp.erase(it + pos);
            }
            for (auto card : temp)
                cards.push_back(card);
            break;
        }
    }
}

template <unsigned n>
bool TripleStraightPlusDouble<n>::operator<(const TripleStraightPlusDouble &rhs) const
{
    return num < rhs.num;
}

template <unsigned n>
bool TripleStraightPlusDouble<n>::operator>(const TripleStraightPlusDouble &rhs) const
{
    return num > rhs.num;
}

template <unsigned n>
bool TripleStraightPlusDouble<n>::isTripleStraightPlusDouble(QVector<Card*> v)
{
    if (v.size() != n * 5)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });

    int cnt[18] = {};
    for (auto card : v)
        cnt[getNum(card)]++;
    for (int i = 15 - n + (n == 1); i >= 3; i--)
    {
        bool flag = true;
        for (int j = i; j < i + int(n); j++)
            if (cnt[j] != 3)
                flag = false;
        if (!flag)
            continue;
        for (int j = 3; j < 18; j++)
        {
            if (j >= i && j < i + int(n))
                continue;
            if (cnt[j] % 2)
                flag = false;
        }
        if (flag)
            return true;
    }
    return false;
}

template <unsigned n>
bool TripleStraightPlusDouble<n>::isLegal(QVector<Card *>v)
{
    if (!isTripleStraightPlusDouble(v))
        return false;
    return *this < TripleStraightPlusDouble<n>(v);
}

#endif // TRIPLESTRAIGHTPLUSDOUBLE_H
