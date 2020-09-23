#include "landlordboard.h"

LandlordBoard::LandlordBoard(int w, int h, QWidget *parent) : QWidget(parent)
{
    setFixedSize(w, h);
}

void LandlordBoard::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    int width = 105 * 3 + 10;
    int l = (400 - width) / 2;
    if (cards.size() == 0)
        for (int i = 0; i < 3; i++)
            p.drawImage(l + 110 * i, 25, QImage(":/pictures/cards/PADDING.png"));
    else
        for (int i = 0; i < 3; i++)
            p.drawImage(l + 110 * i, 25, *(cards[i]->card));
}
