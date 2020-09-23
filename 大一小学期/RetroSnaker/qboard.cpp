#include "qboard.h"

#include <QPainter>
#include <QTime>
#include <QtGlobal>
#include <QDebug>
#include <QMouseEvent>
#include <QVector>
#include <QFontDatabase>

QBoard::QBoard(QWidget *parent) : QFrame(parent)
{
    qsrand(QTime(0,0,0).secsTo(QTime::currentTime()));
    setFixedSize(gridSize * 20 + 2, gridSize * 20 + 2);
    start();
}

void QBoard::start()
{
    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
            grids[i][j] = VOID;
    quint8 x = qrand() % (gridSize - 2) + 1;
    quint8 y = qrand() % (gridSize - 2) + 1;
    dir = qrand() % 4;
    nextDir = 4;
    grids[x][y] = HEAD;
    snake.clear();
    snake.append(QPair<quint8, quint8>(x, y));
    grids[x - dx[dir]][y - dy[dir]] = BODY;
    snake.append(QPair<quint8, quint8>(x - dx[dir], y - dy[dir]));
    grow = time = 0;
}

void QBoard::paintEvent(QPaintEvent *)
{
    QPainter p(this);
    QPen pen(Qt::black, 1);
    QBrush brush(Qt::white);
    p.setPen(pen);
    p.setBrush(brush);
    p.drawRect(0, 0, gridSize * 20 + 1, gridSize * 20 + 1);
    p.setPen(Qt::NoPen);
    QRadialGradient rG;
    QPainterPath path;
    path.moveTo(-8, 10);
    path.lineTo(-8, 2);
    path.arcTo(-8, -6, 16, 16, 180.0, -180.0);
    path.lineTo(8, 10);
    path.lineTo(-8, 10);
    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
        {
            quint16 x = i * 20 + 1, y = j * 20 + 1;
            switch (grids[i][j])
            {
                case HINDER:
                    rG.setCenter(x + 10, y + 10);
                    rG.setRadius(30);
                    rG.setFocalPoint(x + 12, y + 5);
                    rG.setColorAt(1.0, QColor(Qt::yellow).darker(150));
                    rG.setColorAt(0.0, Qt::white);
                    p.setBrush(rG);
                    p.drawRoundedRect(x, y, 20, 20, 5, 5);
                    break;
                case HEAD:
                    brush.setColor(Qt::green);
                    p.setBrush(brush);
                    p.save();
                    switch (dir)
                    {
                        case 0:
                            p.translate(x + 10, y + 10);
                            p.rotate(90);
                            p.drawPath(path);
                            p.restore();
                            brush.setColor(Qt::black);
                            p.setBrush(brush);
                            p.drawEllipse(x + 10, y + 2, 6, 6);
                            p.drawEllipse(x + 10, y + 12, 6, 6);
                            brush.setColor(Qt::white);
                            p.setBrush(brush);
                            p.drawRoundedRect(x + 2, y + 5, 3, 10, 1.5, 1.5);
                            break;
                        case 1:
                            p.translate(x + 10, y + 10);
                            p.drawPath(path);
                            p.restore();
                            brush.setColor(Qt::black);
                            p.setBrush(brush);
                            p.drawEllipse(x + 2, y + 4, 6, 6);
                            p.drawEllipse(x + 12, y + 4, 6, 6);
                            brush.setColor(Qt::white);
                            p.setBrush(brush);
                            p.drawRoundedRect(x + 5, y + 15, 10, 3, 1.5, 1.5);
                            break;
                        case 2:
                            p.translate(x + 10, y + 10);
                            p.rotate(-90);
                            p.drawPath(path);
                            p.restore();
                            brush.setColor(Qt::black);
                            p.setBrush(brush);
                            p.drawEllipse(x + 4, y + 2, 6, 6);
                            p.drawEllipse(x + 4, y + 12, 6, 6);
                            brush.setColor(Qt::white);
                            p.setBrush(brush);
                            p.drawRoundedRect(x + 15, y + 5, 3, 10, 1.5, 1.5);
                            break;
                        case 3:
                            p.translate(x + 10, y + 10);
                            p.rotate(180);
                            p.drawPath(path);
                            p.restore();
                            brush.setColor(Qt::black);
                            p.setBrush(brush);
                            p.drawEllipse(x + 2, y + 10, 6, 6);
                            p.drawEllipse(x + 12, y + 10, 6, 6);
                            brush.setColor(Qt::white);
                            p.setBrush(brush);
                            p.drawRoundedRect(x + 5, y + 2, 10, 3, 1.5, 1.5);
                            break;
                    }
                    break;
                case FOOD:
                    brush.setColor(Qt::red);
                    p.setBrush(brush);
                    p.drawEllipse(x, y, 20, 20);
                    break;
            }
        }
    QPainterPath turn, stra;
    stra.moveTo(-8, -10);
    stra.lineTo(8, -10);
    stra.lineTo(8, 10);
    stra.lineTo(-8, 10);
    stra.lineTo(-8, -10);

    turn.moveTo(-8, -10);
    turn.lineTo(8, -10);
    turn.arcTo(-28, -28, 36, 36, 0, -90);
    turn.lineTo(-10, -8);
    turn.arcTo(-12, -12, 4, 4, 270, 90);

    brush.setColor(Qt::green);
    p.setBrush(brush);

    for (auto body = snake.begin() + 1; body != snake.end(); body++)
    {
        auto pre = body - 1, next = body + 1;
        quint8 x1 = (*pre).first, y1 = (*pre).second;
        quint8 x2 = (*body).first, y2 = (*body).second;
        quint8 d1 = 4;
        quint16 x = x2 * 20 + 1, y = y2 * 20 + 1;
        if (x1 - x2 == 1) d1 = 0;
        if (y2 - y1 == 1) d1 = 1;
        if (x2 - x1 == 1) d1 = 2;
        if (y1 - y2 == 1) d1 = 3;
        if (next == snake.end())
        {
            p.save();
            p.translate(x + 10, y + 10);
            switch (d1)
            {
                case 0:
                    p.rotate(-90);
                    break;
                case 1:
                    p.rotate(180);
                    break;
                case 2:
                    p.rotate(90);
                    break;
            }
            p.drawPath(path);
            p.restore();
            continue;
        }
        quint8 x3 = (*next).first, y3 = (*next).second;
        quint8 d2 = 4;
        if (x2 - x3 == 1) d2 = 0;
        if (y3 - y2 == 1) d2 = 1;
        if (x3 - x2 == 1) d2 = 2;
        if (y2 - y3 == 1) d2 = 3;

        if (d1 == d2)
        {
            p.save();
            p.translate(x + 10, y + 10);
            if (d1 == 0 || d1 == 2)
                p.rotate(90);
            p.drawPath(stra);
            p.restore();
            continue;
        }

        if (d1 == 0 && d2 == 1)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(180);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 0 && d2 == 3)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(90);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 1 && d2 == 0)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 1 && d2 == 2)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(90);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 2 && d2 == 1)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(-90);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 2 && d2 == 3)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 3 && d2 == 0)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(-90);
            p.drawPath(turn);
            p.restore();
            continue;
        }
        if (d1 == 3 && d2 == 2)
        {
            p.save();
            p.translate(x + 10, y + 10);
            p.rotate(180);
            p.drawPath(turn);
            p.restore();
            continue;
        }
    }

    if (status == STOP)
    {
        brush.setColor(QColor(200, 200, 200, 50));
        p.setBrush(brush);
        p.drawRect(0, 0, gridSize * 20 + 1, gridSize * 20 + 1);
        brush.setColor(QColor(0, 0, 0));
        p.setBrush(brush);
        p.setPen(pen);
        p.setFont(QFont(QStringLiteral("华文琥珀"), 70, QFont::Bold));
        p.setRenderHint(QPainter::Antialiasing);
        QFontMetrics metrics = p.fontMetrics();
        quint16 stringHeight = metrics.ascent() + metrics.descent();
        quint16 stringWidth = metrics.horizontalAdvance(QStringLiteral("暂停中"));
        quint16 x = (gridSize * 20 + 1 - stringWidth) / 2;
        if (gridSize * 20 + 1 < stringWidth)
            x = 0;
        quint16 y = (gridSize * 20 + 1 - stringHeight) / 2 + metrics.ascent();
        p.drawText(x, y, QStringLiteral("暂停中"));
    }
    if (status == END)
    {
        brush.setColor(QColor(200, 200, 200, 50));
        p.setBrush(brush);
        p.drawRect(0, 0, gridSize * 20 + 1, gridSize * 20 + 1);
        brush.setColor(QColor(0, 0, 0));
        p.setBrush(brush);
        p.setPen(pen);
        p.setFont(QFont(QStringLiteral("华文琥珀"), 70, QFont::Bold));
        p.setRenderHint(QPainter::Antialiasing);
        QFontMetrics metrics = p.fontMetrics();
        quint16 stringHeight = metrics.ascent() + metrics.descent();
        quint16 stringWidth = metrics.horizontalAdvance(QStringLiteral("游戏结束"));
        quint16 x = (gridSize * 20 + 1 - stringWidth) / 2;
        if (gridSize * 20 + 1 < stringWidth)
            x = 0;
        quint16 y = (gridSize * 20 + 1 - stringHeight) / 2 + metrics.ascent();
        p.drawText(x, y, QStringLiteral("游戏结束"));

        p.setFont(QFont(QStringLiteral("华文楷书"), 20, QFont::Bold));
        metrics = p.fontMetrics();
        stringHeight = metrics.ascent() + metrics.descent();
        stringWidth = metrics.horizontalAdvance(QStringLiteral("请点击“重新开始”"));
        x = (gridSize * 20 + 1 - stringWidth) / 2;
        if (gridSize * 20 + 1 < stringWidth)
            x = 0;
        y = (gridSize * 20 + 1 - stringHeight) / 2 + metrics.ascent();
        p.drawText(x, y + 100, QStringLiteral("请点击“重新开始”"));
    }
}

void QBoard::mouseReleaseEvent(QMouseEvent *ev)
{
    QPoint p = ev->pos();
    if (p.x() == 0 || p.y() == 0 || p.x() == gridSize * 20 + 1 || p.y() == gridSize * 20 + 1 || status != BEFORE_BEGIN)
    {
        QFrame::mouseReleaseEvent(ev);
        return;
    }
    quint8 x = (p.x() - 1) / 20, y = (p.y() - 1) / 20;
    if (grids[x][y] == HEAD || grids[x][y] == BODY)
    {
        QFrame::mouseReleaseEvent(ev);
        return;
    }
    grids[x][y] = VOID + HINDER - grids[x][y];
    update();
}

void QBoard::addFood()
{
    QVector<QPair<quint8, quint8>> v;
    v.clear();
    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
            if (grids[i][j] == VOID)
                v.push_back(QPair<quint8, quint8>(i, j));
    quint16 pos = qrand() % v.length();
    grids[v[pos].first][v[pos].second] = FOOD;
}

bool QBoard::legalPosition(int x, int y)
{
    return x >= 0 && y >= 0 && x < gridSize && y < gridSize;
}
