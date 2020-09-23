#ifndef LANDLORDBOARD_H
#define LANDLORDBOARD_H

#include "card.h"

#include <QWidget>
#include <QPainter>

class LandlordBoard : public QWidget
{
    Q_OBJECT
    friend class GameWindow;
    friend class PackageManager;
public:
    explicit LandlordBoard(int w, int h, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent*);

private:
    QVector<Card*> cards;
};

#endif // LANDLORDBOARD_H
