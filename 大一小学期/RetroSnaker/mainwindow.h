#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "qboard.h"

#include <QMainWindow>
#include <QTimer>
#include <QLabel>
#include <QSettings>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void begin();
    void stop();
    void cont();
    void rebeg();
    void exit();
    void save();
    void load();
    void move();
    void about();
    void generate();

private:
    Ui::MainWindow *ui;
    QBoard *board;
    QTimer *timer;
    QLabel *showTime;
    QLabel *showScore;
    QLabel *showMaxScore;
    static QSettings settings;

    void beforeBeginStatus();
    void gamingStatus();
    void stopStatus();
    void endStatus();
    void keyPressEvent(QKeyEvent*);
};
#endif // MAINWINDOW_H
