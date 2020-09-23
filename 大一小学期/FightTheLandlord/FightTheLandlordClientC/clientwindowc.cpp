#include "clientwindowc.h"
#include "ui_clientwindowc.h"
#include "packagemanager.h"

#include <QGraphicsBlurEffect>
#include <QMessageBox>
#include <QInputDialog>

ClientWindowC::ClientWindowC(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::ClientWindowC)
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

ClientWindowC::~ClientWindowC()
{
    delete ui;
    delete readWriteSocket;
}

void ClientWindowC::on_beginButton_clicked()
{
    ui->beginButton->setDisabled(true);
    connectHost();
}

void ClientWindowC::connectHost()
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

void ClientWindowC::start()
{
    gameWindow->show();
    this->hide();
}

void ClientWindowC::on_connectIP_triggered()
{
    bool flag;
    QString info = QInputDialog::getText(this, QStringLiteral("输入主机 IP"), QStringLiteral("请输入主机 IP地址:"), QLineEdit::Normal, "", &flag);
    if (flag && !info.isEmpty())
        host = QHostAddress(info);
}
