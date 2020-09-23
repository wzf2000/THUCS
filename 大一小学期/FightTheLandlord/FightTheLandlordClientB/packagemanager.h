#ifndef PACKAGEMANAGER_H
#define PACKAGEMANAGER_H

#include "cardsmanager.h"

#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>
#include <QDataStream>

class GameWindow;

class PackageManager
{
public:
    static PackageManager &instance();
    void sendPackage(GameWindow *gameWindow, QTcpSocket *writeSocket, ushort type);
    int readPackage(GameWindow *gameWindow, QTcpSocket *readSocket, int &out);

private:
    static PackageManager _instance;
    QByteArray restBuffer;
    ushort minSize;

    PackageManager();
    ~PackageManager();

    PackageManager(const PackageManager&) = delete;
    void operator=(const PackageManager&) = delete;

    QVector<Card*> getCard(QString, int, GameWindow*);
};

#endif // PACKAGEMANAGER_H
