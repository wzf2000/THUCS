#ifndef CARDBOARD_H
#define CARDBOARD_H

#include "cardsmanager.h"

#include <QWidget>
#include <QPainter>
#include <QEvent>
#include <QMouseEvent>

#define OTHERS 0
#define SELF 1
#define END 2

class CardBoard : public QWidget
{
    Q_OBJECT

    friend class GameWindow;
    friend class PackageManager;
public:
    explicit CardBoard(int w, int h, QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent*);

signals:

private:
    CardsManager *manager;
    int pos = -1;
    int status = OTHERS;
    int w, h;

    void mousePressEvent(QMouseEvent*);
    void mouseReleaseEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent*);

};

#endif // CARDBOARD_H
