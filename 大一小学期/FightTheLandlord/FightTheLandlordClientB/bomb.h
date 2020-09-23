#ifndef BOMB_H
#define BOMB_H

#include "play.h"

class JokerBomb;

class Bomb : public Play
{
public:
    Bomb(const QVector<Card*>&);
    bool operator<(const Bomb&) const;
    bool operator>(const Bomb&) const;
    bool operator<(const JokerBomb&) const;
    bool operator>(const JokerBomb&) const;
    static bool isBomb(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

#endif // BOMB_H
