#ifndef CLIENTWINDOWC_H
#define CLIENTWINDOWC_H

#include "gamewindow.h"

#include <QMainWindow>
#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>

QT_BEGIN_NAMESPACE
namespace Ui { class ClientWindowC; }
QT_END_NAMESPACE

class ClientWindowC : public QMainWindow
{
    Q_OBJECT

    friend class GameWindow;
public:
    ClientWindowC(QWidget *parent = nullptr);
    ~ClientWindowC();

    void connectHost();
    void start();

private slots:
    void on_beginButton_clicked();

    void on_connectIP_triggered();

private:
    Ui::ClientWindowC *ui;
    QTcpSocket *readWriteSocket;
    QHostAddress host = QHostAddress::LocalHost;
    GameWindow *gameWindow = nullptr;
};
#endif // CLIENTWINDOWC_H
