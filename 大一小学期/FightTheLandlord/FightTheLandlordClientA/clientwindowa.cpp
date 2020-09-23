#include "clientwindowa.h"
#include "ui_clientwindowa.h"
#include "packagemanager.h"

#include <QGraphicsBlurEffect>

ClientWindowA::ClientWindowA(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ClientWindowA)
{
    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon/FightTheLandlord.ico"));
    setWindowTitle(QStringLiteral("斗地主"));

    QGraphicsDropShadowEffect *shadow = new QGraphicsDropShadowEffect(this);
    shadow->setOffset(0, 0);
    shadow->setColor(Qt::gray);
    shadow->setBlurRadius(30);
    this->setGraphicsEffect(shadow);
    ui->horizontalLayout->setMargin(10);

    QGraphicsDropShadowEffect *shadowEffect = new QGraphicsDropShadowEffect(this);
    shadowEffect->setOffset(1, 1);
    shadowEffect->setColor(QColor(0, 0, 0, 128));
    shadowEffect->setBlurRadius(20);

    ui->title->setGraphicsEffect(shadowEffect);
}

ClientWindowA::~ClientWindowA()
{
    delete ui;
    delete listenSocket;
}

void ClientWindowA::on_beginButton_clicked()
{
    ui->beginButton->setDisabled(true);
    initServer();
}

void ClientWindowA::initServer()
{
    num = 0;
    listenSocket = new QTcpServer(this);
    listenSocket->listen(QHostAddress::Any, 7777);
    connect(listenSocket, &QTcpServer::newConnection, this, &ClientWindowA::acceptConnection);
}

void ClientWindowA::acceptConnection()
{
    if ((++num) == 2)
        start();
}

void ClientWindowA::start()
{
    gameWindow = new GameWindow(listenSocket, this);
    gameWindow->show();
    this->hide();
}
