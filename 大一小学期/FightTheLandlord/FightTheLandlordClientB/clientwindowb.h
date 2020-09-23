#ifndef CLIENTWINDOWB_H
#define CLIENTWINDOWB_H

#include "gamewindow.h"

#include <QMainWindow>
#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>

QT_BEGIN_NAMESPACE
namespace Ui { class ClientWindowB; }
QT_END_NAMESPACE

class ClientWindowB : public QMainWindow
{
    Q_OBJECT

    friend class GameWindow;
public:
    ClientWindowB(QWidget *parent = nullptr);
    ~ClientWindowB();

    void connectHost();
    void start();

private slots:
    void on_beginButton_clicked();

    void on_connectIP_triggered();

private:
    Ui::ClientWindowB *ui;
    QTcpSocket *readWriteSocket;
    QHostAddress host = QHostAddress::LocalHost;
    GameWindow *gameWindow = nullptr;
};
#endif // CLIENTWINDOWB_H
