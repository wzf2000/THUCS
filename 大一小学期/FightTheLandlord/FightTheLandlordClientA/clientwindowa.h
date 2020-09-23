#ifndef CLIENTWINDOWA_H
#define CLIENTWINDOWA_H

#include "gamewindow.h"

#include <QMainWindow>
#include <QtNetwork/QTcpServer>
#include <QtNetwork/QTcpSocket>
#include <QtNetwork/QHostAddress>

QT_BEGIN_NAMESPACE
namespace Ui { class ClientWindowA; }
QT_END_NAMESPACE

class ClientWindowA : public QMainWindow
{
    Q_OBJECT

    friend class GameWindow;
public:
    ClientWindowA(QWidget *parent = nullptr);
    ~ClientWindowA();

    void initServer();

private slots:
    void on_beginButton_clicked();
    void acceptConnection();

private:
    int num = 0;
    Ui::ClientWindowA *ui;
    QTcpServer *listenSocket = nullptr;
    GameWindow *gameWindow = nullptr;

    void start();
};
#endif // CLIENTWINDOWA_H
