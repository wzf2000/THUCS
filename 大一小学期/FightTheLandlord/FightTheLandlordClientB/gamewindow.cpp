#include "gamewindow.h"
#include "ui_gamewindow.h"
#include "clientwindowb.h"
#include "ui_clientwindowb.h"

#include <QMessageBox>
#include <QThread>
#include <QPainter>
#include <QFontDatabase>
#include <QGraphicsDropShadowEffect>

GameWindow::GameWindow(QTcpSocket *_rw, ClientWindowB *cw, QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::GameWindow)
    , clientWindow(cw)
    , readWriteSocket(_rw)
    , board(new CardBoard(700, 200))
    , outBoard(new CardBoard(700, 200))
    , landlordBoard(new LandlordBoard(400, 200))
    , restShowLeft(new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>17</font>", this))
    , restShowRight(new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>17</font>", this))
    , leftIdentity(new QLabel(""))
    , selfIdentity(new QLabel(""))
    , rightIdentity(new QLabel(""))
    , leftNot(new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>"))
    , selfNot(new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>"))
    , rightNot(new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>"))
    , req(Play::nullPlay)
{
    landlordBoard->cards.clear();
    rest[0] = rest[1] = rest[2] = 17;

    ui->setupUi(this);
    setWindowIcon(QIcon(":/icon/FightTheLandlord.ico"));
    setWindowTitle(QStringLiteral("斗地主"));
    setAttribute(Qt::WA_DeleteOnClose);

    connect(readWriteSocket, &QTcpSocket::readyRead, this, &GameWindow::readPackage);

    ui->horizontalLayout_2->insertWidget(3, selfNot, 0, Qt::AlignCenter);

    leftNot->hide();
    selfNot->hide();
    rightNot->hide();

    QHBoxLayout *hLayout1 = new QHBoxLayout;
    hLayout1->addStretch(3);
    hLayout1->addWidget(selfIdentity);
    hLayout1->addStretch(1);
    hLayout1->addWidget(board);
    hLayout1->addStretch(3);

    ui->verticalLayout->addLayout(hLayout1);

    QHBoxLayout *hLayout2 = new QHBoxLayout;
    hLayout2->addStretch(1);
    hLayout2->addWidget(leftIdentity);
    hLayout2->addStretch(1);
    hLayout2->addWidget(restShowLeft);
    hLayout2->addStretch(1);
    hLayout2->addWidget(leftNot);
    hLayout2->addStretch(3);
    hLayout2->addWidget(outBoard);
    hLayout2->addStretch(3);
    hLayout2->addWidget(rightNot);
    hLayout2->addStretch(1);
    hLayout2->addWidget(restShowRight);
    hLayout2->addStretch(1);
    hLayout2->addWidget(rightIdentity);
    hLayout2->addStretch(1);

    ui->verticalLayout->insertLayout(2, hLayout2);

    ui->verticalLayout->insertWidget(1, landlordBoard, 0, Qt::AlignCenter);

    QGraphicsDropShadowEffect *shadowEffect1 = new QGraphicsDropShadowEffect(this);
    shadowEffect1->setOffset(1, 1);
    shadowEffect1->setColor(QColor(255, 255, 255, 128));
    shadowEffect1->setBlurRadius(10);
    QGraphicsDropShadowEffect *shadowEffect2 = new QGraphicsDropShadowEffect(this);
    shadowEffect2->setOffset(1, 1);
    shadowEffect2->setColor(QColor(255, 255, 255, 128));
    shadowEffect2->setBlurRadius(10);
    QGraphicsDropShadowEffect *shadowEffect3 = new QGraphicsDropShadowEffect(this);
    shadowEffect3->setOffset(1, 1);
    shadowEffect3->setColor(QColor(255, 255, 255, 128));
    shadowEffect3->setBlurRadius(10);
    QGraphicsDropShadowEffect *shadowEffect4 = new QGraphicsDropShadowEffect(this);
    shadowEffect4->setOffset(1, 1);
    shadowEffect4->setColor(QColor(255, 255, 255, 128));
    shadowEffect4->setBlurRadius(10);
    QGraphicsDropShadowEffect *shadowEffect5 = new QGraphicsDropShadowEffect(this);
    shadowEffect5->setOffset(1, 1);
    shadowEffect5->setColor(QColor(255, 255, 255, 128));
    shadowEffect5->setBlurRadius(10);

    restShowLeft->setGraphicsEffect(shadowEffect1);
    restShowRight->setGraphicsEffect(shadowEffect2);
    leftIdentity->setGraphicsEffect(shadowEffect3);
    selfIdentity->setGraphicsEffect(shadowEffect4);
    rightIdentity->setGraphicsEffect(shadowEffect5);
}

GameWindow::~GameWindow()
{
    readWriteSocket->close();
    delete ui;
}

void GameWindow::readPackage()
{
    int flag;
    do
    {
        int out;
        while (!(flag = PackageManager::instance().readPackage(this, readWriteSocket, out)))
            QThread::sleep(1);
        if (out < 2)
        {
            gaming = (gaming + 1) % 3;
            this->out = out;
            if (out == 1)
            {
                pre = (gaming + 2) % 3;
                if (gaming % 3 == LOCALID)
                {
                    leftNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + (!getLandlord ? QStringLiteral("叫地主") : QStringLiteral("抢地主")) + "</font>");
                    leftNot->show();
                }
                if ((gaming + 1) % 3 == LOCALID)
                {
                    rightNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + (!getLandlord ? QStringLiteral("叫地主") : QStringLiteral("抢地主")) + "</font>");
                    rightNot->show();
                }
                getLandlord = true;
            }
            else
            {
                if (gaming % 3 == LOCALID)
                {
                    leftNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + (!getLandlord ? QStringLiteral("不叫") : QStringLiteral("不抢")) + "</font>");
                    leftNot->show();
                }
                if ((gaming + 1) % 3 == LOCALID)
                {
                    rightNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + (!getLandlord ? QStringLiteral("不叫") : QStringLiteral("不抢")) + "</font>");
                    rightNot->show();
                }
            }
            if (gaming == randBegin)
            {
                gaming = landlord = pre;
                pre = -1;
                rest[gaming] += 3;

                if (gaming == (LOCALID + 2) % 3)
                    restShowLeft->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[gaming]) + "</font>");
                if (gaming == (LOCALID + 1) % 3)
                    restShowRight->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[gaming]) + "</font>");

                leftNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
                leftNot->hide();
                selfNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
                selfNot->hide();
                rightNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
                rightNot->hide();

                if (gaming == LOCALID)
                {
                    board->status = SELF;
                    ui->playButton->setDisabled(true);
                    ui->notOutButton->setDisabled(true);
                    ui->playButton->show();
                    ui->notOutButton->show();
                }
                else
                {
                    board->status = OTHERS;
                    ui->playButton->hide();
                    ui->notOutButton->hide();
                }
                setLandlord(gaming);
                continue;
            }
            if (gaming == LOCALID)
            {
                board->status = SELF;

                if (getLandlord)
                {
                    ui->playButton->setText(QStringLiteral("抢地主"));
                    ui->notOutButton->setText(QStringLiteral("不抢"));
                }
                else
                {
                    ui->playButton->setText(QStringLiteral("叫地主"));
                    ui->notOutButton->setText(QStringLiteral("不叫"));
                }

                ui->playButton->setEnabled(true);
                ui->notOutButton->setEnabled(true);
                ui->playButton->show();
                ui->notOutButton->show();
            }
        }
        if (out == 4)
        {
            PackageManager::instance().sendPackage(this, readWriteSocket, 3);
            clientWindow->start();
            ui->playButton->hide();
            ui->notOutButton->hide();
        }
        if (out == 7)
        {
            board->status = OTHERS;
            update();
            landlordBoard->cards.clear();
            landlordBoard->update();
            outBoard->manager->cards.clear();
            outBoard->update();
            req = Play::nullPlay;
            gaming = 0;
            getLandlord = false;
            landlord = pre = -1;
            out = 0;
            rest[0] = rest[1] = rest[2] = 17;
            ui->playButton->hide();
            ui->notOutButton->hide();
            restShowLeft->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>17</font>");
            restShowRight->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>17</font>");
            leftIdentity->setText("");
            selfIdentity->setText("");
            rightIdentity->setText("");
            leftNot->hide();
            selfNot->hide();
            rightNot->hide();
        }
        if (out > 1 && out < 4)
        {
            gaming = (gaming + 1) % 3;
            if ((gaming + 1) % 3 == LOCALID)
                leftNot->hide();
            if ((gaming + 2) % 3 == LOCALID)
                rightNot->hide();
            if (gaming % 3 == LOCALID)
                selfNot->hide();
            if (out == 2)
            {
                if (pre == gaming)
                {
                    pre = -1;
                    req = Play::nullPlay;
                }
                this->out = 0;
                if (gaming == LOCALID)
                    leftNot->show();
                if ((gaming + 1) % 3 == LOCALID)
                    rightNot->show();
            }
            else
            {
                this->out = 1;
                pre = (gaming + 2) % 3;
                rest[pre] -= req->cards.size();
                if (pre == (LOCALID + 2) % 3)
                    restShowLeft->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[pre]) + "</font>");
                if (pre == (LOCALID + 1) % 3)
                    restShowRight->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[pre]) + "</font>");
                if (!rest[pre])
                {
                    board->status = END;
                    update();
                    pre = -1;
                    ui->playButton->setText(QStringLiteral("再玩一次"));
                    ui->notOutButton->setText(QStringLiteral("退出"));
                    ui->playButton->show();
                    ui->notOutButton->show();
                    ui->playButton->setEnabled(true);
                    ui->notOutButton->setEnabled(true);
                    continue;
                }
            }

            if (gaming == LOCALID)
            {
                board->status = SELF;
                ui->playButton->show();
                ui->notOutButton->show();
                ui->playButton->setDisabled(true);
                if (pre == -1)
                    ui->notOutButton->setDisabled(true);
                else
                    ui->notOutButton->setEnabled(true);
            }
            outBoard->manager->cards = req->cards;
            outBoard->update();
        }
        if (out == 13)
        {
            ui->playButton->setDisabled(true);
            QMessageBox::warning(this, QStringLiteral("连接断开"), QStringLiteral("连接已断开，请点击退出！"));
        }
        if (out > 15 && out < 19)
        {
            randBegin = gaming = pre = out - 16;
            board->status = (LOCALID == gaming);
            this->out = 0;
            if (out == 16 + LOCALID)
            {
                ui->playButton->setText(QStringLiteral("叫地主"));
                ui->notOutButton->setText(QStringLiteral("不叫"));
                ui->playButton->show();
                ui->notOutButton->show();
                ui->playButton->setEnabled(true);
                ui->notOutButton->setEnabled(true);
            }
            else
            {
                ui->playButton->hide();
                ui->notOutButton->hide();
            }

        }
    } while (flag == 2);
}

void GameWindow::on_playButton_clicked()
{
    if (board->status == END)
    {
        PackageManager::instance().sendPackage(this, readWriteSocket, 4);
        ui->playButton->setDisabled(true);
        return;
    }
    if (landlord == -1)
    {
        pre = LOCALID;
        gaming = (gaming + 1) % 3;
        out = 1;
        getLandlord = true;

        selfNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + ui->playButton->text() + "</font>");
        selfNot->show();
        ui->playButton->setText(QStringLiteral("出牌"));
        ui->notOutButton->setText(QStringLiteral("不出"));
        ui->playButton->hide();
        ui->notOutButton->hide();
        PackageManager::instance().sendPackage(this, readWriteSocket, 1);
        board->status = OTHERS;

        if (gaming == randBegin)
        {
            gaming = landlord = LOCALID;
            board->status = SELF;
            out = 0;
            randBegin = pre = -1;
            rest[gaming] += 3;
            ui->playButton->setDisabled(true);
            ui->playButton->setDisabled(true);
            ui->playButton->show();
            ui->notOutButton->show();
            leftNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            leftNot->hide();
            selfNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            selfNot->hide();
            rightNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            rightNot->hide();
            setLandlord(gaming);
        }
        return;
    }
    QVector<Card*> &v = outBoard->manager->cards;
    QVector<Card*> tmp;
    v.clear();
    tmp.clear();
    for (auto card : board->manager->cards)
        if (card->chosen)
            v.push_back(card);
        else
            tmp.push_back(card);
    if (req->id)
        delete req;
    switch (v.size())
    {
        case 1:
            req = new Single(v[0]);
            break;
        case 2:
            if (Double::isDouble(v))
                req = new Double(v[0], v[1]);
            else
                req = new JokerBomb(v[0], v[1]);
            break;
        case 3:
            req = new Triple<1>(v);
            break;
        case 4:
            if (Bomb::isBomb(v))
                req = new Bomb(v);
            else
                req = new TripleStraightPlusSingle<1>(v);
            break;
        case 5:
            if (Straight<5>::isStraight(v))
                req = new Straight<5>(v);
            else
                req = new TripleStraightPlusDouble<1>(v);
            break;
        case 6:
            if (Straight<6>::isStraight(v))
                req = new Straight<6>(v);
            else
                if (DoubleStraight<3>::isDoubleStraight(v))
                    req = new DoubleStraight<3>(v);
                else
                    if (Triple<2>::isTriple(v))
                        req = new Triple<2>(v);
                    else
                        req = new QuartetPlusTwoSingle(v);
            break;
        case 7:
            req = new Straight<7>(v);
            break;
        case 8:
            if (Straight<8>::isStraight(v))
                req = new Straight<8>(v);
            else
                if (DoubleStraight<4>::isDoubleStraight(v))
                    req = new DoubleStraight<4>(v);
                else
                    if (QuartetPlusTwoDouble::isQuartetPlusTwoDouble(v))
                        req = new QuartetPlusTwoDouble(v);
                    else
                        req = new TripleStraightPlusSingle<2>(v);
            break;
        case 9:
            if (Triple<3>::isTriple(v))
                req = new Triple<3>(v);
            else
                req = new Straight<9>(v);
            break;
        case 10:
            if (Straight<10>::isStraight(v))
                req = new Straight<10>(v);
            else
                if (DoubleStraight<5>::isDoubleStraight(v))
                    req = new DoubleStraight<5>(v);
                else
                    req = new TripleStraightPlusDouble<2>(v);
            break;
        case 11:
            req = new Straight<11>(v);
            break;
        case 12:
            if (Straight<12>::isStraight(v))
                req = new Straight<12>(v);
            else
                if (DoubleStraight<6>::isDoubleStraight(v))
                    req = new DoubleStraight<6>(v);
                else
                    if (Triple<4>::isTriple(v))
                        req = new Triple<4>(v);
                    else
                        req = new TripleStraightPlusSingle<3>(v);
            break;
        case 14:
            req = new DoubleStraight<7>(v);
            break;
        case 15:
            if (Triple<5>::isTriple(v))
                req = new Triple<5>(v);
            else
                req = new TripleStraightPlusDouble<3>(v);
            break;
        case 16:
            if (DoubleStraight<8>::isDoubleStraight(v))
                req = new DoubleStraight<8>(v);
            else
                req = new TripleStraightPlusSingle<4>(v);
            break;
        case 18:
            if (DoubleStraight<9>::isDoubleStraight(v))
                req = new DoubleStraight<9>(v);
            else
                req = new Triple<6>(v);
            break;
        case 20:
            if (DoubleStraight<10>::isDoubleStraight(v))
                req = new DoubleStraight<10>(v);
            else
                if (TripleStraightPlusSingle<5>::isTripleStraightPlusSingle(v))
                    req = new TripleStraightPlusSingle<5>(v);
                else
                    req = new TripleStraightPlusDouble<4>(v);
            break;
    }
    v = req->cards;
    outBoard->update();
    board->manager->cards = tmp;
    board->update();
    board->status = OTHERS;
    rest[LOCALID] -= v.size();

    out = 1;

    PackageManager::instance().sendPackage(this, readWriteSocket, 2);

    if (!rest[LOCALID])
    {
        board->status = END;
        update();
        pre = -1;
        ui->playButton->setText(QStringLiteral("再玩一次"));
        ui->notOutButton->setText(QStringLiteral("退出"));
        ui->playButton->setEnabled(true);
        ui->notOutButton->setEnabled(true);
        return;
    }

    gaming = (gaming + 1) % 3;
    pre = LOCALID;
    ui->playButton->hide();
    ui->notOutButton->hide();
    rightNot->hide();
}

void GameWindow::on_notOutButton_clicked()
{
    if (board->status == END)
    {
        PackageManager::instance().sendPackage(this, readWriteSocket, 8);
        clientWindow->show();
        clientWindow->ui->beginButton->setEnabled(true);
        close();
        return;
    }
    if (landlord == -1)
    {
        gaming = (gaming + 1) % 3;
        out = 0;

        selfNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>" + ui->notOutButton->text() + "</font>");
        selfNot->show();
        ui->playButton->setText(QStringLiteral("出牌"));
        ui->notOutButton->setText(QStringLiteral("不出"));
        ui->playButton->hide();
        ui->notOutButton->hide();
        PackageManager::instance().sendPackage(this, readWriteSocket, 1);
        board->status = OTHERS;

        if (gaming == randBegin)
        {
            gaming = landlord = pre;
            board->status = OTHERS;
            out = 0;
            randBegin = pre = -1;
            rest[gaming] += 3;

            leftNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            leftNot->hide();
            selfNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            selfNot->hide();
            rightNot->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;' color=yellow>不出</font>");
            rightNot->hide();

            if (gaming == (LOCALID + 2) % 3)
                restShowLeft->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[gaming]) + "</font>");
            if (gaming == (LOCALID + 1) % 3)
                restShowRight->setText("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("剩余牌数: ") + "</font><font style='font-size:30px;font-weight:700;'>" + QString::number(rest[gaming]) + "</font>");
            setLandlord(gaming);
        }
        return;
    }
    for (auto card : board->manager->cards)
        card->chosen = false;
    board->update();
    board->status = OTHERS;

    out = 0;

    PackageManager::instance().sendPackage(this, readWriteSocket, 2);

    gaming = (gaming + 1) % 3;
    if (pre == gaming)
    {
        pre = -1;
        req = Play::nullPlay;
        outBoard->manager->cards = req->cards;
        outBoard->update();
    }

    ui->playButton->hide();
    ui->notOutButton->hide();
    selfNot->show();
    rightNot->hide();
}

void GameWindow::setLandlord(int pos)
{
    QFont font(QStringLiteral("华文行楷"), 30);
    leftIdentity->setFont(font);
    rightIdentity->setFont(font);
    selfIdentity->setFont(font);

    leftIdentity->setWordWrap(true);
    leftIdentity->setAlignment(Qt::AlignTop);
    rightIdentity->setWordWrap(true);
    rightIdentity->setAlignment(Qt::AlignTop);

    leftIdentity->setTextFormat(Qt::MarkdownText);
    rightIdentity->setTextFormat(Qt::MarkdownText);
    selfIdentity->setTextFormat(Qt::MarkdownText);

    leftIdentity->setText(QStringLiteral("# 农民"));
    rightIdentity->setText(QStringLiteral("# 农民"));
    selfIdentity->setText(QStringLiteral("# 农民"));

    if (pos == LOCALID)
        selfIdentity->setText(QStringLiteral("# 地主"));
    if ((pos + 1) % 3 == LOCALID)
        leftIdentity->setText(QStringLiteral("# 地主"));
    if ((pos + 2) % 3 == LOCALID)
        rightIdentity->setText(QStringLiteral("# 地主"));
}

void GameWindow::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    if (board->status == END)
    {
        bool flag = false;
        if (rest[landlord] == 0 && landlord == LOCALID)
            flag = true;
        if (rest[landlord] != 0 && landlord != LOCALID)
            flag = true;
        QPen pen(Qt::white);
        QBrush brush(Qt::black);

        p.setBrush(brush);
        p.setPen(pen);
        p.setFont(QFont(QStringLiteral("华文行楷"), 50, QFont::Bold));
        p.setRenderHint(QPainter::Antialiasing);
        QFontMetrics metrics = p.fontMetrics();
        int stringHeight = metrics.ascent() + metrics.descent();
        int stringWidth = metrics.horizontalAdvance(flag ? QStringLiteral("胜利") : QStringLiteral("失败"));
        int x = (rect().width() - stringWidth) / 2;
        if (rect().width() < stringWidth)
            x = 0;
        int y = (rect().height() - stringHeight) / 2 + metrics.ascent();
        p.drawText(x + 300, y - 220, flag ? QStringLiteral("胜利") : QStringLiteral("失败"));
    }
}
