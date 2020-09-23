#include "clientwindowc.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ClientWindowC w;
    w.show();
    return a.exec();
}
