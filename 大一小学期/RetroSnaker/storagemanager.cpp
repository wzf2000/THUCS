#include "storagemanager.h"

#include <QDebug>

StorageManager StorageManager::_instance;

StorageManager::StorageManager()
{

}

StorageManager &StorageManager::instance()
{
    return _instance;
}

bool StorageManager::input(const QString &path, QBoard *board)
{
    file = new QFile(path);
    if (!file->open(QIODevice::ReadOnly))
    {
        delete file;
        return false;
    }

    QDataStream aStream(file);
    aStream.setByteOrder(QDataStream::LittleEndian);

    QString copyright;
    char* buf;
    uint strLen;
    aStream.readBytes(buf, strLen);
    copyright = QString::fromLocal8Bit(buf, strLen);

    if (copyright != "@wzf2000")
    {
        delete file;
        return false;
    }

    quint32 time;
    quint16 grow;
    quint8 dir, grids[gridSize][gridSize];

    aStream.readRawData((char *)&time, sizeof(quint32));

    aStream.readRawData((char *)&dir, sizeof(quint8));
    if (dir > 3)
    {
        delete file;
        return false;
    }

    aStream.readRawData((char *)&grow, sizeof(quint16));
    if (grow > gridSize * gridSize)
    {
        delete file;
        return false;
    }

    aStream.readRawData((char *)grids, sizeof(quint8) * gridSize * gridSize);
    qint16 cnt = 0;
    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
        {
            if (grids[i][j] > 4)
            {
                delete file;
                return false;
            }
            if (grids[i][j] == HEAD)
                if (cnt++)
                {
                    delete file;
                    return false;
                }
        }

    quint16 len;
    aStream.readRawData((char *)&len, sizeof(quint16));
    QList<QPair<quint8, quint8>> snake;
    snake.clear();

    bool isSnake[gridSize][gridSize] = {};

    quint8 preX, preY;

    for (quint16 i = 0; i < len; i++)
    {
        quint8 x, y;
        aStream.readRawData((char *)&x, sizeof(quint8));
        aStream.readRawData((char *)&y, sizeof(quint8));
        if (i && abs(x - preX) + abs(y - preY) != 1)
        {
            delete file;
            return false;
        }
        snake.append(QPair<quint8, quint8>(x, y));
        if (!board->legalPosition(x, y) || grids[x][y] != (i == 0 ? HEAD : BODY))
        {
            delete file;
            return false;
        }
        isSnake[x][y] = true;
        preX = x, preY = y;
    }

    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
            if (!isSnake[i][j] && (grids[i][j] == HEAD || grids[i][j] == BODY))
            {
                delete file;
                return false;
            }

    board->time = time;
    board->dir = dir;
    board->grow = grow;
    board->snake = snake;

    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
            board->grids[i][j] = grids[i][j];

    file->close();
    delete file;
    file = nullptr;

    return true;
}

bool StorageManager::output(const QString &path, QBoard *board)
{
    QFileInfo fileInfo(path);
    if (fileInfo.exists())
    {
        file = new QFile(path);
        file->remove();
        delete file;
        file = nullptr;
    }

    file = new QFile(path);
    if (!file->open(QIODevice::WriteOnly))
    {
        delete file;
        return false;
    }

    QDataStream aStream(file);
    aStream.setByteOrder(QDataStream::LittleEndian);

    QString copyright("@wzf2000");
    QByteArray btArray = copyright.toUtf8();
    aStream.writeBytes(btArray, btArray.length());

    aStream.writeRawData((char *)&board->time, sizeof(quint32));
    aStream.writeRawData((char *)&board->dir, sizeof(quint8));
    aStream.writeRawData((char *)&board->grow, sizeof(quint16));
    aStream.writeRawData((char *)board->grids, sizeof(quint8) * gridSize * gridSize);

    quint16 len = board->snake.length();

    aStream.writeRawData((char *)&len, sizeof(quint16));

    for (auto el : board->snake)
    {
        aStream.writeRawData((char *)&el.first, sizeof(quint8));
        aStream.writeRawData((char *)&el.second, sizeof(quint8));
    }

    file->close();
    delete file;
    file = nullptr;

    return true;
}
