#include "clientwindowa.h"

#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ClientWindowA w;
    w.show();
    return a.exec();
}
