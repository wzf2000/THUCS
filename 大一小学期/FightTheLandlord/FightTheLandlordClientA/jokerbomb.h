#ifndef JOKERBOMB_H
#define JOKERBOMB_H

#include "bomb.h"

class JokerBomb : public Play
{
public:
    JokerBomb(const Card*, const Card*);
    JokerBomb(const QVector<Card*>&);
    bool operator<(const Bomb&) const;
    bool operator>(const Bomb&) const;
    static bool isJokerBomb(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;
};

#endif // JOKERBOMB_H
