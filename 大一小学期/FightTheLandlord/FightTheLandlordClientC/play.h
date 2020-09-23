#ifndef PLAY_H
#define PLAY_H

#define NULLPLAY 0
#define SINGLE 1
#define DOUBLE 2
#define TRIPLE 3
#define STRAIGHT 4
#define DOUBLESTRAIGHT 5
#define TRIPLESTRAIGHTPLUSSINGLE 6
#define TRIPLESTRAIGHTPLUSDOUBLE 7
#define BOMB 8
#define JOKERBOMB 9
#define QUARTEPLUSTWOSINGLE 10
#define QUARTEPLUSTWODOUBLE 11

#include "card.h"

class Play
{
    friend class GameWindow;
    friend class PackageManager;
public:
    Play(int, int, QVector<Card*>);
    Play(int, int);
    virtual ~Play();

    virtual bool isLegal(QVector<Card*>);
    static int getNum(Card*);

    static Play* nullPlay;
    int id;
    int N;

protected:
    QVector<Card*> cards;
};

#endif // PLAY_H
