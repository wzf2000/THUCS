#ifndef QUARTETPLUSTWODOUBLE_H
#define QUARTETPLUSTWODOUBLE_H

#include "play.h"

class QuartetPlusTwoDouble : public Play
{
public:
    QuartetPlusTwoDouble(const QVector<Card*>&);
    bool operator<(const QuartetPlusTwoDouble&) const;
    bool operator>(const QuartetPlusTwoDouble&) const;
    static bool isQuartetPlusTwoDouble(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

#endif // QUARTETPLUSTWODOUBLE_H
