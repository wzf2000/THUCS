#ifndef TRIPLESTRAIGHTPLUSSINGLE_H
#define TRIPLESTRAIGHTPLUSSINGLE_H

#include "play.h"

template <unsigned n>
class TripleStraightPlusSingle : public Play
{
public:
    TripleStraightPlusSingle(const QVector<Card*>&);
    bool operator<(const TripleStraightPlusSingle<n>&) const;
    bool operator>(const TripleStraightPlusSingle<n>&) const;
    static bool isTripleStraightPlusSingle(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

template <unsigned n>
TripleStraightPlusSingle<n>::TripleStraightPlusSingle(const QVector<Card*> &v) : Play(TRIPLESTRAIGHTPLUSSINGLE, n, v)
{
    std::sort(cards.begin(), cards.end(), [](const Card *l, const Card *r) { return *l < *r; });

    int cnt[16] = {};
    for (auto card : cards)
        if (card->num != 1 && card->num != 2)
            cnt[card->num]++;
        else
            cnt[card->num + 13]++;
    for (int i = 15 - n + (n == 1); i >= 3; i--)
    {
        bool flag = true;
        for (int j = i; j < i + int(n); j++)
            if (cnt[j] < 3)
                flag = false;
        if (flag)
        {
            num = i;
            QVector<Card*> temp(cards);
            cards.clear();
            QVector<Card*>::iterator it = temp.begin();
            for (int j = i; j < i + int(n); j++)
            {
                while (getNum(*it) != j)
                    it++;
                cards.push_back(*it);
                temp.erase(it);
                cards.push_back(*it);
                temp.erase(it);
                cards.push_back(*it);
                temp.erase(it);
            }
            for (auto card : temp)
                cards.push_back(card);
            break;
        }
    }
}

template <unsigned n>
bool TripleStraightPlusSingle<n>::operator<(const TripleStraightPlusSingle &rhs) const
{
    return num < rhs.num;
}

template <unsigned n>
bool TripleStraightPlusSingle<n>::operator>(const TripleStraightPlusSingle &rhs) const
{
    return num > rhs.num;
}

template <unsigned n>
bool TripleStraightPlusSingle<n>::isTripleStraightPlusSingle(QVector<Card*> v)
{
    if (v.size() != n * 4)
        return false;
    std::sort(v.begin(), v.end(), [](const Card *l, const Card *r) { return *l < *r; });

    int cnt[18] = {};
    for (auto card : v)
        cnt[getNum(card)]++;
    for (int i = 15 - n + (n == 1); i >= 3; i--)
    {
        bool flag = true;
        for (int j = i; j < i + int(n); j++)
            if (cnt[j] < 3)
                flag = false;
        if (flag)
            return true;
    }
    return false;
}

template <unsigned n>
bool TripleStraightPlusSingle<n>::isLegal(QVector<Card *>v)
{
    if (!isTripleStraightPlusSingle(v))
        return false;
    return *this < TripleStraightPlusSingle<n>(v);
}

#endif // TRIPLESTRAIGHTPLUSSINGLE_H
