#include "packagemanager.h"
#include "gamewindow.h"

//  1: 0/1 -> Seizing/not
//  2: 0/1 -> play/not out + out type + cards
//  3: started 0/1/2 -> A/B/C
//  4: restart 0/1/2 -> A/B/C
//  5: 17~33 card
//  6: 34~50 card
//  7: 51~53 card
//  8: exit 0/1/2 -> A/B/C
//  9: begin 0/1/2 -> A/B/C
// 10: landloard cards

PackageManager PackageManager::_instance;

PackageManager::PackageManager()
{
    restBuffer.clear();
    QByteArray byte;
    QDataStream out(&byte, QIODevice::WriteOnly);
    out << ushort(0) << ushort(0);
    minSize = byte.size();
}

PackageManager::~PackageManager()
{

}

PackageManager &PackageManager::instance()
{
    return _instance;
}

void PackageManager::sendPackage(GameWindow *gameWindow, QTcpSocket *writeSocket, ushort type)
{
    QByteArray sendByte;
    QDataStream out(&sendByte, QIODevice::WriteOnly);
    out.setByteOrder(QDataStream::BigEndian);
    out << ushort(0) << ushort(0);

    QString info;

    switch (type)
    {
        case 1:
            out << ushort(gameWindow->out);
            break;
        case 2:
            out << ushort(gameWindow->out);
            if (gameWindow->out)
            {
                out << ushort(gameWindow->req->id) << ushort(gameWindow->req->N);
                info = "";
                for (auto card : gameWindow->req->cards)
                    info += card->getName().leftJustified(11);
                out << info;
            }
            break;
        case 3:
            out << ushort(LOCALID);
            break;
        case 4:
            out << ushort(LOCALID);
            break;
        case 5:
            info = "";
            for (int i = 17; i < 34; i++)
                info += gameWindow->v[i]->getName().leftJustified(11);
            out << info;
            break;
        case 6:
            info = "";
            for (int i = 34; i < 51; i++)
                info += gameWindow->v[i]->getName().leftJustified(11);
            out << info;
            break;
        case 7:
            info = "";
            for (int i = 51; i < 54; i++)
                info += gameWindow->v[i]->getName().leftJustified(11);
            out << info;
            break;
        case 8:
            out << ushort(LOCALID);
            break;
        case 9:
            out << ushort(gameWindow->gaming);
            break;
        case 10:
            info = "";
            for (int i = 51; i < 54; i++)
                info += gameWindow->v[i]->getName().leftJustified(11);
            out << info;
            break;
    }

    out.device()->seek(0);
    ushort len = ushort(sendByte.size());
    out << type << len;
    writeSocket->write(sendByte);
}

QVector<Card*> PackageManager::getCard(QString info, int num, GameWindow *window)
{
    QVector<Card*> v;
    int pos = 0;
    v.clear();
    for (int i = 0; i < num; i++, pos += 11)
    {
        if (info[pos] == 'R' && info[pos + 1] == 'E')
            v.push_back(new Card('R', window));
        else
            if (info[pos] == 'B' && info[pos + 1] == 'L')
                v.push_back(new Card('B', window));
            else
                if (info[pos + 1] == '1' && info[pos + 2] == '0')
                    v.push_back(new Card(info[pos].toLatin1(), 10, window));
                else
                    if (info[pos + 1] == 'A')
                        v.push_back(new Card(info[pos].toLatin1(), 1, window));
                    else
                        if (info[pos + 1] == 'J')
                            v.push_back(new Card(info[pos].toLatin1(), 11, window));
                        else
                            if (info[pos + 1] == 'Q')
                                v.push_back(new Card(info[pos].toLatin1(), 12, window));
                            else
                                if (info[pos + 1] == 'K')
                                    v.push_back(new Card(info[pos].toLatin1(), 13, window));
                                else
                                    v.push_back(new Card(info[pos].toLatin1(), info[pos + 1].toLatin1() - '0', window));
    }
    return v;
}

int PackageManager::readPackage(GameWindow *gameWindow, QTcpSocket *readSocket, int &out)
{
    if (readSocket->bytesAvailable() + restBuffer.size() <= 0)
        return 0;
    QByteArray buffer;
    buffer = readSocket->readAll();
    restBuffer.append(buffer);

    ushort type, len, tmp, id, N;
    QString info;
    int totalLen = restBuffer.size();

    QDataStream packet(restBuffer);
    QVector<Card*> v;
    packet.setByteOrder(QDataStream::BigEndian);

    if (totalLen < minSize)
        return 0;
    packet >> type >> len;
    if (totalLen < len)
        return 0;

    switch (type)
    {
        case 1:
            packet >> tmp;
            out = int(tmp);
            break;
        case 2:
            packet >> tmp;
            out = int(tmp) + 2;
            if (out == 2)
                break;
            packet >> id >> N;
            switch (id)
            {
                case SINGLE:
                    packet >> info;
                    gameWindow->req = new Single(getCard(info, 1, gameWindow));
                    break;
                case DOUBLE:
                    packet >> info;
                    gameWindow->req = new Double(getCard(info, 2, gameWindow));
                    break;
                case TRIPLE:
                    packet >> info;
                    switch (N)
                    {
                        case 1:
                            gameWindow->req = new Triple<1>(getCard(info, N * 3, gameWindow));
                            break;
                        case 2:
                            gameWindow->req = new Triple<2>(getCard(info, N * 3, gameWindow));
                            break;
                        case 3:
                            gameWindow->req = new Triple<3>(getCard(info, N * 3, gameWindow));
                            break;
                        case 4:
                            gameWindow->req = new Triple<4>(getCard(info, N * 3, gameWindow));
                            break;
                        case 5:
                            gameWindow->req = new Triple<5>(getCard(info, N * 3, gameWindow));
                            break;
                        case 6:
                            gameWindow->req = new Triple<6>(getCard(info, N * 3, gameWindow));
                            break;
                    }
                    break;
                case STRAIGHT:
                    packet >> info;
                    switch (N)
                    {
                        case 5:
                            gameWindow->req = new Straight<5>(getCard(info, N, gameWindow));
                            break;
                        case 6:
                            gameWindow->req = new Straight<6>(getCard(info, N, gameWindow));
                            break;
                        case 7:
                            gameWindow->req = new Straight<7>(getCard(info, N, gameWindow));
                            break;
                        case 8:
                            gameWindow->req = new Straight<8>(getCard(info, N, gameWindow));
                            break;
                        case 9:
                            gameWindow->req = new Straight<9>(getCard(info, N, gameWindow));
                            break;
                        case 10:
                            gameWindow->req = new Straight<10>(getCard(info, N, gameWindow));
                            break;
                        case 11:
                            gameWindow->req = new Straight<11>(getCard(info, N, gameWindow));
                            break;
                        case 12:
                            gameWindow->req = new Straight<12>(getCard(info, N, gameWindow));
                            break;
                    }
                    break;
                case DOUBLESTRAIGHT:
                    packet >> info;
                    switch (N)
                    {
                        case 3:
                            gameWindow->req = new DoubleStraight<3>(getCard(info, N * 2, gameWindow));
                            break;
                        case 4:
                            gameWindow->req = new DoubleStraight<4>(getCard(info, N * 2, gameWindow));
                            break;
                        case 5:
                            gameWindow->req = new DoubleStraight<5>(getCard(info, N * 2, gameWindow));
                            break;
                        case 6:
                            gameWindow->req = new DoubleStraight<6>(getCard(info, N * 2, gameWindow));
                            break;
                        case 7:
                            gameWindow->req = new DoubleStraight<7>(getCard(info, N * 2, gameWindow));
                            break;
                        case 8:
                            gameWindow->req = new DoubleStraight<8>(getCard(info, N * 2, gameWindow));
                            break;
                        case 9:
                            gameWindow->req = new DoubleStraight<9>(getCard(info, N * 2, gameWindow));
                            break;
                        case 10:
                            gameWindow->req = new DoubleStraight<10>(getCard(info, N * 2, gameWindow));
                            break;
                    }
                    break;
                case TRIPLESTRAIGHTPLUSSINGLE:
                    packet >> info;
                    switch (N)
                    {
                        case 1:
                            gameWindow->req = new TripleStraightPlusSingle<1>(getCard(info, N * 4, gameWindow));
                            break;
                        case 2:
                            gameWindow->req = new TripleStraightPlusSingle<2>(getCard(info, N * 4, gameWindow));
                            break;
                        case 3:
                            gameWindow->req = new TripleStraightPlusSingle<3>(getCard(info, N * 4, gameWindow));
                            break;
                        case 4:
                            gameWindow->req = new TripleStraightPlusSingle<4>(getCard(info, N * 4, gameWindow));
                            break;
                        case 5:
                            gameWindow->req = new TripleStraightPlusSingle<5>(getCard(info, N * 4, gameWindow));
                            break;
                    }
                    break;
                case TRIPLESTRAIGHTPLUSDOUBLE:
                    packet >> info;
                    switch (N)
                    {
                        case 1:
                            gameWindow->req = new TripleStraightPlusDouble<1>(getCard(info, N * 5, gameWindow));
                            break;
                        case 2:
                            gameWindow->req = new TripleStraightPlusDouble<2>(getCard(info, N * 5, gameWindow));
                            break;
                        case 3:
                            gameWindow->req = new TripleStraightPlusDouble<3>(getCard(info, N * 5, gameWindow));
                            break;
                        case 4:
                            gameWindow->req = new TripleStraightPlusDouble<4>(getCard(info, N * 5, gameWindow));
                            break;
                    }
                    break;
                case BOMB:
                    packet >> info;
                    gameWindow->req = new Bomb(getCard(info, 4, gameWindow));
                    break;
                case JOKERBOMB:
                    packet >> info;
                    gameWindow->req = new JokerBomb(getCard(info, 2, gameWindow));
                    break;
                case QUARTEPLUSTWOSINGLE:
                    packet >> info;
                    gameWindow->req = new QuartetPlusTwoSingle(getCard(info, 6, gameWindow));
                    break;
                case QUARTEPLUSTWODOUBLE:
                    packet >> info;
                    gameWindow->req = new QuartetPlusTwoDouble(getCard(info, 8, gameWindow));
                    break;
            }
            gameWindow->outBoard->manager->cards = gameWindow->req->cards;
            gameWindow->outBoard->update();
            break;
        case 3:
            packet >> tmp;
            out = int(tmp) + 4;
            break;
        case 4:
            packet >> tmp;
            out = int(tmp) + 7;
            break;
        case 5:
            out = 10;
            packet >> info;
            gameWindow->board->manager->cards = getCard(info, 17, gameWindow);
            std::sort(gameWindow->board->manager->cards.begin(), gameWindow->board->manager->cards.end(), [](const Card *l, const Card *r) { return *l > *r; });
            gameWindow->board->update();
            break;
        case 6:
            out = 11;
            packet >> info;
            gameWindow->board->manager->cards = getCard(info, 17, gameWindow);
            std::sort(gameWindow->board->manager->cards.begin(), gameWindow->board->manager->cards.end(), [](const Card *l, const Card *r) { return *l > *r; });
            gameWindow->board->update();
            break;
        case 7:
            out = 12;
            packet >> info;
            v = getCard(info, 3, gameWindow);
            gameWindow->board->manager->cards.append(v);
            std::sort(gameWindow->board->manager->cards.begin(), gameWindow->board->manager->cards.end(), [](const Card *l, const Card *r) { return *l > *r; });
            gameWindow->board->update();
            break;
        case 8:
            packet >> tmp;
            out = int(tmp) + 13;
            break;
        case 9:
            packet >> tmp;
            out = int(tmp) + 16;
            break;
        case 10:
            out = 19;
            packet >> info;
            gameWindow->landlordBoard->cards = getCard(info, 3, gameWindow);
            gameWindow->landlordBoard->update();
            break;
    }

    buffer = restBuffer.right(totalLen - len);
    totalLen = buffer.size();
    restBuffer = buffer;

    if (totalLen < minSize)
        return 1;

    QDataStream pack(restBuffer);
    pack >> type >> len;
    if (totalLen < len)
        return 1;

    return 2;
}
