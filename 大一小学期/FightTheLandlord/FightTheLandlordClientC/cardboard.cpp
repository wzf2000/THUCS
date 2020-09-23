#include "cardboard.h"
#include "gamewindow.h"
#include "ui_gamewindow.h"

#include <QDebug>

CardBoard::CardBoard(int _w, int _h, QWidget *parent) : QWidget(parent), manager(new CardsManager), w(_w), h(_h)
{
    setFixedSize(w, h);
}

void CardBoard::paintEvent(QPaintEvent *)
{
    int sz = manager->cards.size();
    int width = sz * 25 + 80;
    QPainter p(this);
    int cnt = 0;
    for (auto card : manager->cards)
    {
        int height = (h - 150) / 2;
        if (card->chosen)
            height -= 20;

        p.drawImage((w - width) / 2 + 25 * (cnt++), height, *(card->card));
        if (card->toBeChosen)
        {
            p.save();
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0, 0, 0, 128));
            p.drawRect((w - width) / 2 + 25 * (cnt - 1), height, 105, 150);
            p.restore();
        }
    }
}

void CardBoard::mousePressEvent(QMouseEvent *ev)
{
    if (status != SELF || static_cast<GameWindow*>(static_cast<QWidget*>(parentWidget())->parentWidget())->landlord == -1)
    {
        QWidget::mousePressEvent(ev);
        return;
    }
    int sz = manager->cards.size();
    int width = sz * 25 + 80;
    int x = ev->x(), y = ev->y();
    int l = (w - width) / 2, r = l + width;
    if (x < l || x >= r)
    {
        QWidget::mousePressEvent(ev);
        return;
    }
    for (int i = 0; i < sz; i++)
    {
        if (x < l + 25 * i || x >= (i < sz - 1 ? l + 25 * (i + 1) : l + 25 * i + 105))
            continue;
        if (manager->cards[i]->chosen)
        {
            if (y >= (h - 150) / 2 - 20 && y < (h + 150) / 2 - 20)
            {
                pos = i;
                manager->cards[i]->toBeChosen = true;
                update();
                return;
            }
            else
            {
                QWidget::mousePressEvent(ev);
                return;
            }
        }
        else
        {
            if (y >= (h - 150) / 2 && y < (h + 150) / 2)
            {
                pos = i;
                manager->cards[i]->toBeChosen = true;
                update();
                return;
            }
            else
            {
                QWidget::mousePressEvent(ev);
                return;
            }
        }
    }
}

void CardBoard::mouseReleaseEvent(QMouseEvent *ev)
{
    if (status != SELF || static_cast<GameWindow*>(static_cast<QWidget*>(parentWidget())->parentWidget())->landlord == -1)
    {
        QWidget::mouseReleaseEvent(ev);
        return;
    }
    QVector<Card*> v;
    for (auto card : manager->cards)
    {
        if (card->toBeChosen)
            card->chosed();
        if (card->chosen)
            v.push_back(card);
    }

    GameWindow *par = static_cast<GameWindow*>(static_cast<QWidget*>(parentWidget())->parentWidget());

    par->ui->notOutButton->setEnabled(true);

    if (par->req->id)
    {
        if (par->req->isLegal(v))
            par->ui->playButton->setEnabled(true);
        else
        {
            if (par->req->id == JOKERBOMB)
                par->ui->playButton->setDisabled(true);
            else
                if (par->req->id == BOMB)
                {
                    if (JokerBomb::isJokerBomb(v))
                        par->ui->playButton->setEnabled(true);
                    else
                        par->ui->playButton->setDisabled(true);
                }
                else
                {
                    if (Bomb::isBomb(v) || JokerBomb::isJokerBomb(v))
                        par->ui->playButton->setEnabled(true);
                    else
                        par->ui->playButton->setDisabled(true);
                }
        }
    }
    else
    {
        par->ui->notOutButton->setDisabled(true);
        int sz = v.size();
        par->ui->playButton->setDisabled(true);
        if (sz == 1)
            par->ui->playButton->setEnabled(true);
        if (sz == 2 && (Double::isDouble(v) || JokerBomb::isJokerBomb(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 3 && Triple<1>::isTriple(v))
            par->ui->playButton->setEnabled(true);
        if (sz == 4 && (Bomb::isBomb(v) || TripleStraightPlusSingle<1>::isTripleStraightPlusSingle(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 5 && (Straight<5>::isStraight(v) || TripleStraightPlusDouble<1>::isTripleStraightPlusDouble(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 6 && (DoubleStraight<3>::isDoubleStraight(v) || Triple<2>::isTriple(v) || Straight<6>::isStraight(v) || QuartetPlusTwoSingle::isQuartetPlusTwoSingle(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 7 && Straight<7>::isStraight(v))
            par->ui->playButton->setEnabled(true);
        if (sz == 8 && (DoubleStraight<4>::isDoubleStraight(v) || Straight<8>::isStraight(v) || QuartetPlusTwoDouble::isQuartetPlusTwoDouble(v) || TripleStraightPlusSingle<2>::isTripleStraightPlusSingle(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 9 && (Triple<3>::isTriple(v) || Straight<9>::isStraight(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 10 && (DoubleStraight<5>::isDoubleStraight(v) || Straight<10>::isStraight(v) || TripleStraightPlusDouble<2>::isTripleStraightPlusDouble(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 11 && Straight<11>::isStraight(v))
            par->ui->playButton->setEnabled(true);
        if (sz == 12 && (DoubleStraight<6>::isDoubleStraight(v) || Triple<4>::isTriple(v) || TripleStraightPlusSingle<3>::isTripleStraightPlusSingle(v) || Straight<12>::isStraight(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 14 && DoubleStraight<7>::isDoubleStraight(v))
            par->ui->playButton->setEnabled(true);
        if (sz == 15 && (Triple<5>::isTriple(v) || TripleStraightPlusDouble<3>::isTripleStraightPlusDouble(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 16 && (DoubleStraight<8>::isDoubleStraight(v) || TripleStraightPlusSingle<4>::isTripleStraightPlusSingle(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 18 && (DoubleStraight<9>::isDoubleStraight(v) || Triple<6>::isTriple(v)))
            par->ui->playButton->setEnabled(true);
        if (sz == 20 && (DoubleStraight<10>::isDoubleStraight(v) || TripleStraightPlusSingle<5>::isTripleStraightPlusSingle(v) || TripleStraightPlusDouble<4>::isTripleStraightPlusDouble(v)))
            par->ui->playButton->setEnabled(true);
    }
    update();
}

void CardBoard::mouseMoveEvent(QMouseEvent *ev)
{
    if (status != SELF || static_cast<GameWindow*>(static_cast<QWidget*>(parentWidget())->parentWidget())->landlord == -1)
    {
        QWidget::mouseMoveEvent(ev);
        return;
    }
    if (pos == -1)
    {
        for (auto card : manager->cards)
            card->toBeChosen = false;
        QWidget::mouseMoveEvent(ev);
        update();
        return;
    }
    int sz = manager->cards.size();
    int width = sz * 25 + 80;
    int x = ev->x(), y = ev->y();
    int l = (w - width) / 2, r = l + width;
    if (x < l || x >= r)
    {
        for (auto card : manager->cards)
            card->toBeChosen = false;
        QWidget::mouseMoveEvent(ev);
        update();
        return;
    }
    for (int i = 0; i < sz; i++)
    {
        if (x < l + 25 * i || x >= (i < sz - 1 ? l + 25 * (i + 1) : l + 25 * i + 105))
            continue;
        if (manager->cards[i]->chosen)
        {
            if (y >= (h - 150) / 2 - 20 && y < (h + 150) / 2 - 20)
            {
                int L = pos, R = i;
                if (L > R)
                    std::swap(L, R);
                for (auto card : manager->cards)
                    card->toBeChosen = false;
                for (int j = L; j <= R; j++)
                    manager->cards[j]->toBeChosen = true;
                update();
                return;
            }
            else
            {
                for (auto card : manager->cards)
                    card->toBeChosen = false;
                QWidget::mouseReleaseEvent(ev);
                update();
                return;
            }
        }
        else
        {
            if (y >= (h - 150) / 2 && y < (h + 150) / 2)
            {
                int L = pos, R = i;
                if (L > R)
                    std::swap(L, R);
                for (auto card : manager->cards)
                    card->toBeChosen = false;
                for (int j = L; j <= R; j++)
                    manager->cards[j]->toBeChosen = true;
                update();
                return;
            }
            else
            {
                for (auto card : manager->cards)
                    card->toBeChosen = false;
                QWidget::mouseReleaseEvent(ev);
                update();
                return;
            }
        }
    }
}
