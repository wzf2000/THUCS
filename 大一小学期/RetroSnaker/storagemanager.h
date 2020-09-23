#ifndef STORAGEMANAGER_H
#define STORAGEMANAGER_H

#include "qboard.h"

#include <QFile>
#include <QFileInfo>

class StorageManager
{
public:
    static StorageManager &instance();

    bool input(const QString&, QBoard*);
    bool output(const QString&, QBoard*);

private:
    QFile *file = nullptr;
    static StorageManager _instance;

    StorageManager();

    StorageManager(const StorageManager&) = delete;
    void operator=(const StorageManager&) = delete;
};

#endif // STORAGEMANAGER_H
