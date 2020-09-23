#ifndef QUARTETPLUSTWOSINGLE_H
#define QUARTETPLUSTWOSINGLE_H

#include "play.h"

class QuartetPlusTwoSingle : public Play
{
public:
    QuartetPlusTwoSingle(const QVector<Card*>&);
    bool operator<(const QuartetPlusTwoSingle&) const;
    bool operator>(const QuartetPlusTwoSingle&) const;
    static bool isQuartetPlusTwoSingle(QVector<Card*>);
    bool isLegal(QVector<Card*>) override;

private:
    int num;
};

#endif // QUARTETPLUSTWOSINGLE_H
