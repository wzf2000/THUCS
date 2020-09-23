#include "play.h"

Play* Play::nullPlay = new Play(NULLPLAY, 0);

Play::Play(int _t, int _n) : id(_t), N(_n)
{
    cards.clear();
}

Play::Play(int _t, int _n, QVector<Card*> v) : id(_t), N(_n), cards(v)
{

}

Play::~Play()
{

}

int Play::getNum(Card *card)
{
    if (card->isJoker)
    {
        if (card->type == 'R')
            return 17;
        else
            return 16;
    }
    else
    {
        if (card->num < 3)
            return card->num + 13;
        else
            return card->num;
    }
}

bool Play::isLegal(QVector<Card*>)
{
    return true;
}
