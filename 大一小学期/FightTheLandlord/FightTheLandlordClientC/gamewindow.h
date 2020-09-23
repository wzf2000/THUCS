#ifndef GAMEWINDOW_H
#define GAMEWINDOW_H

#include "card.h"
#include "cardboard.h"
#include "packagemanager.h"
#include "landlordboard.h"

#include <QMainWindow>
#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>
#include <QLabel>

#define LOCALID 2

namespace Ui { class GameWindow; }

class ClientWindowC;

class GameWindow : public QMainWindow
{
    Q_OBJECT

    friend class PackageManager;
    friend class CardBoard;
public:
    GameWindow(QTcpSocket *_rw, ClientWindowC *cw, QWidget *parent = nullptr);
    ~GameWindow();

protected:
    void paintEvent(QPaintEvent*);

private slots:
    void on_playButton_clicked();
    void on_notOutButton_clicked();
    void readPackage();

private:
    Ui::GameWindow *ui;
    ClientWindowC *clientWindow;
    QTcpSocket *readWriteSocket = nullptr;
    CardBoard *board, *outBoard;
    LandlordBoard *landlordBoard;
    QVector<Card*> v;
    QLabel *restShowLeft, *restShowRight;
    QLabel *leftIdentity, *selfIdentity, *rightIdentity;
    QLabel *leftNot, *selfNot, *rightNot;
    Play *req;
    int gaming = 0;
    int pre = -1;
    bool out = 0;
    int rest[3];
    int landlord = -1;
    int randBegin = -1;
    bool getLandlord = false;

    void setLandlord(int);
};

#endif // GAMEWINDOW_H
