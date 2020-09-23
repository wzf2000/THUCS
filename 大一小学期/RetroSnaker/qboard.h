#ifndef QBOARD_H
#define QBOARD_H

#include <QObject>
#include <QWidget>
#include <QFrame>
#include <QList>
#include <QPair>

#define BEFORE_BEGIN 0
#define GAMING 1
#define STOP 2
#define END 3

#define VOID 0
#define HINDER 1
#define HEAD 2
#define BODY 3
#define FOOD 4

const quint8 gridSize = 40;
const quint32 frame = 75;
const qint8 dx[4] = {1, 0, -1, 0};
const qint8 dy[4] = {0, -1, 0, 1};
// right up left down

class QBoard : public QFrame
{
    Q_OBJECT
    friend class MainWindow;
    friend class StorageManager;
public:
    QBoard(QWidget *parent = nullptr);

protected:
    void paintEvent(QPaintEvent*);

private:
    quint8 grids[gridSize][gridSize] = {};
    quint8 status = BEFORE_BEGIN;
    quint8 dir = 0;
    quint8 nextDir = 4;
    quint16 grow = 0;
    quint32 time = 0;
    QList<QPair<quint8, quint8>> snake;

    void mouseReleaseEvent(QMouseEvent*);
    void start();
    void addFood();
    bool legalPosition(int, int);
};

#endif // QBOARD_H
