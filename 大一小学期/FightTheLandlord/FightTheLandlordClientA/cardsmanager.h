#ifndef CARDSMANAGER_H
#define CARDSMANAGER_H

#include "card.h"
#include "play.h"
#include "bomb.h"
#include "double.h"
#include "doublestraight.h"
#include "jokerbomb.h"
#include "quartetplustwosingle.h"
#include "quartetplustwodouble.h"
#include "single.h"
#include "straight.h"
#include "triple.h"
#include "triplestraightplussingle.h"
#include "triplestraightplusdouble.h"

#include <QWidget>
#include <QVector>

class CardsManager : public QWidget
{
    Q_OBJECT

    friend class CardBoard;
    friend class GameWindow;
    friend class PackageManager;
public:
    explicit CardsManager(QWidget *parent = nullptr);

signals:

private:
    QVector<Card*> cards;

};

/*
 *  1: 1
 *  2: 2 or joker * 2
 *  3: 3
 *  4: 3 + 1 or 4
 *  5: 3 + 2 or 1 * 5
 *  6: 2 * 3 or 3 + 3 or 1 * 7 or 4 + 1 * 2
 *  7: 1 * 7
 *  8: 2 * 4 or (3 + 1) * 2 or 1 * 8 or 4 + 2 * 2
 *  9: 3 * 3 or 1 * 9
 * 10: 2 * 5 or (3 + 2) * 2 or 1 * 10
 * 11: 1 * 11
 * 12: 2 * 6 or 3 * 4 or (3 + 1) * 3 or 1 * 12
 * 13:
 * 14: 2 * 7
 * 15: 3 * 5 or (3 + 2) * 3
 * 16: 2 * 8 or (3 + 1) * 4
 * 17:
 * 18: 2 * 9 or 3 * 6
 * 19:
 * 20: 2 * 10 or (3 + 1) * 5 or (3 + 2) * 4
 */

#endif // CARDSMANAGER_H
