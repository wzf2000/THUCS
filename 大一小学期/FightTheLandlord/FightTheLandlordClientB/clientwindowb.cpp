#include "clientwindowb.h"
#include "ui_clientwindowb.h"
#include "packagemanager.h"

#include <QGraphicsBlurEffect>
#include <QMessageBox>
#include <QInputDialog>

ClientWindowB::ClientWindowB(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ClientWindowB)
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

ClientWindowB::~ClientWindowB()
{
    delete ui;
    delete readWriteSocket;
}

void ClientWindowB::on_beginButton_clicked()
{
    ui->beginButton->setDisabled(true);
    connectHost();
}

void ClientWindowB::connectHost()
{
    readWriteSocket = new QTcpSocket(this);
    readWriteSocket->connectToHost(host, 7777);
    gameWindow = new GameWindow(readWriteSocket, this);
    if (!readWriteSocket->waitForConnected(60000))
    {
        readWriteSocket->close();
        delete readWriteSocket;
        ui->beginButton->setEnabled(true);
        QMessageBox::warning(this, QStringLiteral("连接失败"), QStringLiteral("连接超时，请重新尝试！"));
    }
}

void ClientWindowB::start()
{
    gameWindow->show();
    this->hide();
}

void ClientWindowB::on_connectIP_triggered()
{
    bool flag;
    QString info = QInputDialog::getText(this, QStringLiteral("输入主机 IP"), QStringLiteral("请输入主机 IP地址:"), QLineEdit::Normal, "", &flag);
    if (flag && !info.isEmpty())
        host = QHostAddress(info);
}
