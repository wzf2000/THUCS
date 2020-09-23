#include "card.h"
#include "single.h"

Card::Card(char tp, QWidget *parent) : QWidget(parent), isJoker(true), type(tp), num(0), chosen(false), toBeChosen(false)
{
    init();
}

Card::Card(char tp, int n, QWidget *parent) : QWidget(parent), isJoker(false), type(tp), num(n), chosen(false), toBeChosen(false)
{
    init();
}

void Card::init()
{
    card = new QImage(":/pictures/cards/" + getName() + ".jpg");
}

QString Card::getName()
{
    if (isJoker)
    {
        switch (type)
        {
            case 'R':
                return "RED JOKER";
                break;
            case 'B':
                return "BLACK JOKER";
                break;
        }
    }
    switch (num)
    {
        case 1:
            return QString(type) + "A";
            break;
        case 11:
            return QString(type) + "J";
            break;
        case 12:
            return QString(type) + "Q";
            break;
        case 13:
            return QString(type) + "K";
            break;
        default:
            return QString(type) + QString::number(num);
    }
}

bool Card::operator<(const Card &rhs) const
{
    if (Single(this) < Single(&rhs)) return true;
    if (Single(this) > Single(&rhs)) return false;
    return type < rhs.type;
}

bool Card::operator>(const Card &rhs) const
{
    if (Single(this) > Single(&rhs)) return true;
    if (Single(this) < Single(&rhs)) return false;
    return type > rhs.type;
}

void Card::chosed()
{
    chosen ^= 1;
    toBeChosen = 0;
}
