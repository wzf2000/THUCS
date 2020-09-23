QT       += core gui
QT       += network

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

CONFIG += c++11

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    card.cpp \
    landlordboard.cpp \
    packagemanager.cpp \
    play.cpp \
    bomb.cpp \
    double.cpp \
    jokerbomb.cpp \
    quartetplustwodouble.cpp \
    quartetplustwosingle.cpp \
    single.cpp \
    cardsmanager.cpp \
    cardboard.cpp \
    gamewindow.cpp \
    clientwindowb.cpp \
    main.cpp

HEADERS += \
    card.h \
    landlordboard.h \
    packagemanager.h \
    play.h \
    bomb.h \
    double.h \
    doublestraight.h \
    jokerbomb.h \
    quartetplustwodouble.h \
    quartetplustwosingle.h \
    single.h \
    straight.h \
    triple.h \
    triplestraightplusdouble.h \
    triplestraightplussingle.h \
    cardsmanager.h \
    cardboard.h \
    gamewindow.h \
    clientwindowb.h

FORMS += \
    clientwindowb.ui \
    gamewindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    resource.qrc

RC_ICONS = FightTheLandlord.ico
