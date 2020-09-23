#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "storagemanager.h"

#include <QToolBar>
#include <QPainter>
#include <QDebug>
#include <QKeyEvent>
#include <QFileDialog>
#include <QMessageBox>
#include <QGraphicsDropShadowEffect>

QSettings MainWindow::settings("HKEY_CURRENT_USER\\Software\\RetroSnake", QSettings::NativeFormat);

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , board(new QBoard(this))
    , timer(new QTimer(this))
    , showTime(new QLabel("<font style='font-size:30px;font-weight:700;'>0</font>", this))
    , showScore(new QLabel("<font style='font-size:30px;font-weight:700;'>0</font>", this))
    , showMaxScore(new QLabel("<font style='font-size:30px;font-weight:700;'>" + QString::number(settings.value("maxScore").toInt()) + "</font>", this))
{
    ui->setupUi(this);
    setWindowTitle(QStringLiteral("贪吃蛇"));
    setWindowIcon(QIcon(":/icon/RetroSnake.ico"));

    QGraphicsDropShadowEffect *shadow = new QGraphicsDropShadowEffect(this);
    shadow->setOffset(0, 0);
    shadow->setColor(Qt::gray);
    shadow->setBlurRadius(30);
    this->setGraphicsEffect(shadow);
    ui->horizontalLayout->setMargin(10);

    timer->stop();

    QHBoxLayout *mainLayout = ui->horizontalLayout;
    mainLayout->addStretch(1);
    mainLayout->addWidget(board);
    mainLayout->addStretch(1);

    QGraphicsDropShadowEffect *shadowEffect1 = new QGraphicsDropShadowEffect(this);
    shadowEffect1->setOffset(1, 1);
    shadowEffect1->setColor(QColor(0, 0, 0, 128));
    shadowEffect1->setBlurRadius(10);

    QGraphicsDropShadowEffect *shadowEffect2 = new QGraphicsDropShadowEffect(this);
    shadowEffect2->setOffset(1, 1);
    shadowEffect2->setColor(QColor(0, 0, 0, 128));
    shadowEffect2->setBlurRadius(10);

    QGraphicsDropShadowEffect *shadowEffect3 = new QGraphicsDropShadowEffect(this);
    shadowEffect3->setOffset(1, 1);
    shadowEffect3->setColor(QColor(0, 0, 0, 128));
    shadowEffect3->setBlurRadius(10);


    QVBoxLayout *showLayout = new QVBoxLayout;
    showLayout->addStretch(3);
    QLabel *showText = new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("当前时间") + ":</font>", this);
    showText->setGraphicsEffect(shadowEffect1);
    showLayout->addWidget(showText);
    showTime->setAlignment(Qt::AlignCenter);
    showLayout->addWidget(showTime);
    showLayout->addStretch(1);
    QLabel *scoreText = new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("当前分数") + ":</font>", this);
    scoreText->setGraphicsEffect(shadowEffect2);
    showLayout->addWidget(scoreText);
    showScore->setAlignment(Qt::AlignCenter);
    showLayout->addWidget(showScore);
    showLayout->addStretch(1);
    QLabel *maxScoreText = new QLabel("<font style='font-family:\"华文行楷\";font-weight:400;font-size:40px;'>" + QStringLiteral("历史最高") + ":</font>", this);
    maxScoreText->setGraphicsEffect(shadowEffect3);
    showLayout->addWidget(maxScoreText);
    showMaxScore->setAlignment(Qt::AlignCenter);
    showLayout->addWidget(showMaxScore);
    showLayout->addStretch(3);

    mainLayout->addLayout(showLayout);
    mainLayout->addStretch(1);

    connect(ui->beg, &QAction::triggered, this, &MainWindow::begin);
    connect(ui->stop, &QAction::triggered, this, &MainWindow::stop);
    connect(ui->cont, &QAction::triggered, this, &MainWindow::cont);
    connect(ui->rebeg, &QAction::triggered, this, &MainWindow::rebeg);
    connect(ui->exit, &QAction::triggered, this, &MainWindow::exit);
    connect(ui->save, &QAction::triggered, this, &MainWindow::save);
    connect(ui->load, &QAction::triggered, this, &MainWindow::load);
    connect(ui->about, &QAction::triggered, this, &MainWindow::about);
    connect(ui->generate, &QAction::triggered, this, &MainWindow::generate);

    connect(ui->begButton, &QPushButton::clicked, this, &MainWindow::begin);
    connect(ui->stopButton, &QPushButton::clicked, this, &MainWindow::stop);
    connect(ui->contButton, &QPushButton::clicked, this, &MainWindow::cont);
    connect(ui->rebegButton, &QPushButton::clicked, this, &MainWindow::rebeg);
    connect(ui->exitButton, &QPushButton::clicked, this, &MainWindow::exit);
    connect(ui->saveButton, &QPushButton::clicked, this, &MainWindow::save);
    connect(ui->loadButton, &QPushButton::clicked, this, &MainWindow::load);

    connect(timer, &QTimer::timeout, this, &MainWindow::move);
}

MainWindow::~MainWindow()
{
    delete ui;
    delete board;
    delete timer;
    delete showTime;
}

void MainWindow::begin()
{
    gamingStatus();
    board->addFood();
    board->update();
}

void MainWindow::stop()
{
    stopStatus();
    board->update();
}

void MainWindow::cont()
{
    gamingStatus();
    board->update();
}

void MainWindow::rebeg()
{
    beforeBeginStatus();
    board->start();
    board->update();
    showTime->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(board->time) + "</font>");
    quint32 score = (board->snake.length() + board->grow) / 3 * 5;
    showScore->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(score) + "</font>");
}

void MainWindow::exit()
{
    QApplication::exit(0);
}

void MainWindow::save()
{
    QString file_name = QFileDialog::getSaveFileName(this, QStringLiteral("保存为格局文件"), ".", tr("Data file(*.dat)"));
    StorageManager &sM = StorageManager::instance();
    if (!sM.output(file_name, board))
        QMessageBox::critical(nullptr, QStringLiteral("错误"), QStringLiteral("保存格局文件失败"), QMessageBox::Ok, QMessageBox::Ok);
}

void MainWindow::load()
{
    QString file_name = QFileDialog::getOpenFileName(this, QStringLiteral("载入格局文件"), ".", tr("Data file(*.dat)"));
    StorageManager &sM = StorageManager::instance();
    if (!sM.input(file_name, board))
        QMessageBox::critical(nullptr, QStringLiteral("错误"), QStringLiteral("载入格局文件失败"), QMessageBox::Ok, QMessageBox::Ok);
    else
    {
        stopStatus();
        board->update();
        showTime->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(board->time) + "</font>");
        quint32 score = (board->snake.length() + board->grow) / 3 * 5;
        quint32 maxScore = settings.value("maxScore").toInt();
        showScore->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(score) + "</font>");
        if (score > maxScore)
        {
            settings.setValue("maxScore", score);
            showMaxScore->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(score) + "</font>");
        }
    }
}

void MainWindow::move()
{
    quint8 pre = board->dir;
    if (board->nextDir != 4)
    {
        board->dir = board->nextDir;
        board->nextDir = 4;
    }

    QPair<quint8, quint8> head = board->snake.front();
    int x = head.first, y = head.second;
    if ((board->grids[x + dx[board->dir]][y + dy[board->dir]] != VOID && board->grids[x + dx[board->dir]][y + dy[board->dir]] != FOOD) || !board->legalPosition(x + dx[board->dir], y + dy[board->dir]))
    {
        endStatus();
        board->dir = pre;
        board->update();
        return;
    }

    bool flag = board->grids[x + dx[board->dir]][y + dy[board->dir]] == FOOD;

    board->grids[x][y] = BODY;
    board->grids[x + dx[board->dir]][y + dy[board->dir]] = HEAD;

    board->snake.push_front(QPair<int, int>(x + dx[board->dir], y + dy[board->dir]));

    if (board->grow)
        board->grow--;
    else
    {
        board->grids[board->snake.back().first][board->snake.back().second] = VOID;
        board->snake.pop_back();
    }
    if (flag)
    {
        board->grow += 3;
        board->addFood();
    }
    board->update();
    board->time++;
    showTime->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(board->time) + "</font>");
    quint32 score = (board->snake.length() + board->grow) / 3 * 5;
    quint32 maxScore = settings.value("maxScore").toInt();
    showScore->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(score) + "</font>");
    if (score > maxScore)
    {
        settings.setValue("maxScore", score);
        showMaxScore->setText("<font style='font-size:30px;font-weight:700;'>" + QString::number(score) + "</font>");
    }
}

void MainWindow::about()
{
    QMessageBox::about(this, QStringLiteral("关于贪吃蛇"), QStringLiteral("<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<strong>贪吃蛇游戏</strong>是一款休闲益智类游戏，既简单又耐玩。该游戏通过控制蛇头方向吃蛋，从而使得蛇变得越来越长。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;在本游戏中，你可以通过<strong>上下左右</strong>或者 <strong>WASD</strong> 控制蛇前进的方向，每个食物会使得蛇在之后的三个单位时间内增长一个单位长度。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;除此之外，在开始游戏前，你还可以<strong>通过点击空白区域使得其转换为墙体</strong>，再次点击也可转回空白。</p><p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;本游戏还设置了存档、读档功能，你可以选择在游戏进行到一定程度后保存游戏格局，在之后继续尝试。</p><h2><strong><center>来挑战你的最高分吧！</center></strong></h2>"));
}

void MainWindow::beforeBeginStatus()
{
    board->status = BEFORE_BEGIN;

    ui->beg->setEnabled(true);
    ui->exit->setEnabled(true);
    ui->load->setEnabled(true);
    ui->stop->setDisabled(true);
    ui->cont->setDisabled(true);
    ui->save->setDisabled(true);
    ui->rebeg->setDisabled(true);
    ui->generate->setEnabled(true);

    ui->begButton->setEnabled(true);
    ui->exitButton->setEnabled(true);
    ui->loadButton->setEnabled(true);
    ui->stopButton->setDisabled(true);
    ui->contButton->setDisabled(true);
    ui->saveButton->setDisabled(true);
    ui->rebegButton->setDisabled(true);

    timer->stop();
}

void MainWindow::gamingStatus()
{
    board->status = GAMING;

    ui->beg->setDisabled(true);
    ui->exit->setEnabled(true);
    ui->load->setDisabled(true);
    ui->stop->setEnabled(true);
    ui->cont->setDisabled(true);
    ui->save->setDisabled(true);
    ui->rebeg->setDisabled(true);
    ui->generate->setDisabled(true);

    ui->begButton->setDisabled(true);
    ui->exitButton->setEnabled(true);
    ui->loadButton->setDisabled(true);
    ui->stopButton->setEnabled(true);
    ui->contButton->setDisabled(true);
    ui->saveButton->setDisabled(true);
    ui->rebegButton->setDisabled(true);

    timer->start(frame);
}

void MainWindow::stopStatus()
{
    board->status = STOP;

    ui->beg->setDisabled(true);
    ui->exit->setEnabled(true);
    ui->load->setDisabled(true);
    ui->stop->setDisabled(true);
    ui->cont->setEnabled(true);
    ui->save->setEnabled(true);
    ui->rebeg->setEnabled(true);
    ui->generate->setDisabled(true);

    ui->begButton->setDisabled(true);
    ui->exitButton->setEnabled(true);
    ui->loadButton->setDisabled(true);
    ui->stopButton->setDisabled(true);
    ui->contButton->setEnabled(true);
    ui->saveButton->setEnabled(true);
    ui->rebegButton->setEnabled(true);

    timer->stop();
}

void MainWindow::endStatus()
{
    board->status = END;

    ui->beg->setDisabled(true);
    ui->exit->setEnabled(true);
    ui->load->setDisabled(true);
    ui->stop->setDisabled(true);
    ui->cont->setDisabled(true);
    ui->save->setDisabled(true);
    ui->rebeg->setEnabled(true);
    ui->generate->setDisabled(true);

    ui->begButton->setDisabled(true);
    ui->exitButton->setEnabled(true);
    ui->loadButton->setDisabled(true);
    ui->stopButton->setDisabled(true);
    ui->contButton->setDisabled(true);
    ui->saveButton->setDisabled(true);
    ui->rebegButton->setEnabled(true);

    timer->stop();
}

void MainWindow::keyPressEvent(QKeyEvent *ev)
{
    if (board->status != GAMING)
    {
        QMainWindow::keyPressEvent(ev);
        return;
    }
    switch (ev->key())
    {
        case Qt::Key_W:
        case Qt::Key_Up:
            if (board->dir != 3)
                board->nextDir = 1;
            break;
        case Qt::Key_D:
        case Qt::Key_Right:
            if (board->dir != 2)
                board->nextDir = 0;
            break;
        case Qt::Key_S:
        case Qt::Key_Down:
            if (board->dir != 1)
                board->nextDir = 3;
            break;
        case Qt::Key_A:
        case Qt::Key_Left:
            if (board->dir != 0)
                board->nextDir = 2;
            break;
        default:
            QMainWindow::keyPressEvent(ev);
    }
}

void MainWindow::generate()
{
    ui->generate->setDisabled(true);
    QVector<QPair<quint8, quint8>> v;
    v.clear();
    for (quint8 i = 0; i < gridSize; i++)
        for (quint8 j = 0; j < gridSize; j++)
            if (board->grids[i][j] == VOID)
                v.push_back(QPair<quint8, quint8>(i, j));
    for (quint16 i = 0; i < 20; i++)
    {
        quint16 pos = qrand() % v.length();
        quint8 x = v[pos].first, y = v[pos].second;
        board->grids[x][y] = HINDER;
        v.remove(pos);
    }
    board->update();
}
