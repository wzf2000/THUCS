#ifndef DOUBLE_H
#define DOUBLE_H

#include "play.h"

class Double : public Play
{
public:
    Double(const Card*, const Card*);
    Double(const QVector<Card*>&);
    bool operator<(const Double&) const;
    bool operator>(const Double&) const;
    static bool isDouble(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

#endif // DOUBLE_H
