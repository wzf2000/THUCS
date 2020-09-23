#ifndef SINGLE_H
#define SINGLE_H

#include "play.h"

class Single : public Play
{
public:
    Single(const Card*);
    Single(const QVector<Card*>&);
    bool operator<(const Single&) const;
    bool operator>(const Single&) const;
    static bool isSingle(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

#endif // SINGLE_H
