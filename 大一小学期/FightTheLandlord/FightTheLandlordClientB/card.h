#ifndef CARD_H
#define CARD_H

#include <QtAlgorithms>
#include <QWidget>
#include <QDebug>

class Card : public QWidget
{
    Q_OBJECT

    friend class Play;
    friend class CardsManager;
    friend class Single;
    friend class Double;
    template <unsigned n>
    friend class Triple;
    template <unsigned n>
    friend class Straight;
    template <unsigned n>
    friend class DoubleStraight;
    friend class Bomb;
    friend class JokerBomb;
    friend class QuartetPlusTwoSingle;
    friend class QuartetPlusTwoDouble;
    template <unsigned n>
    friend class TripleStraightPlusSingle;
    template <unsigned n>
    friend class TripleStraightPlusDouble;
    friend class CardBoard;
    friend class LandlordBoard;
    friend class GameWindow;
public:
    explicit Card(char tp, QWidget *parent = nullptr);
    explicit Card(char tp, int n, QWidget *parent = nullptr);
    bool operator<(const Card&) const;
    bool operator>(const Card&) const;

    void chosed();
    QString getName();

signals:

private:
    bool isJoker;
    char type;
    int num;
    bool chosen;
    bool toBeChosen;
    QImage *card = nullptr;

    void init();

};

#endif // CARD_H
