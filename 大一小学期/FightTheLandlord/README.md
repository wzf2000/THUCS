<h2><center>网络斗地主设计文档</center></h2>

<p style="text-align:right"><strong>2019011200 计 93 王哲凡</strong></p>

---

### 1. 项目基本结构

#### 1.1. 预览

其中 `ClientA` 作为接受其他客户端连接请求的客户端，结构如下。

![预览1](https://img.wzf2000.top/image/2020/09/23/1.png)

![预览2](https://img.wzf2000.top/image/2020/09/23/2.png)

`ClientB` 和 `ClientC` 的结构也类似，只是对应窗口文件（`clientwindow*.cpp` 和 `gamewindow.cpp`）实现有区别。

#### 1.2. 项目简介

项目主要由窗口部分、卡牌类、牌型类、牌管理器类、卡牌显示控件、包管理器类和 UI 文件、资源文件组成。

三个客户端的实现仅在窗口部分和 UI 文件有所区别。

每个客户端均有两个窗口即 `ClientWindow*` 和 `GameWindow`，其中 `ClientWindow*` 负责准备界面（在 `ClientB` 和 `ClientC` 中还包括设置主机 IP 部分），`GameWindow` 负责游戏界面以及网络通信。

本项目的三个客户端通过 AB 连接和 AC 连接实现任意两个客户端的通信，通过在 `gamewindow.h` 中定义不同的宏 `LOCALID` 在网络通信中区分不同的客户端。

而卡牌类（`Card`）主要负责卡牌大小判断，图片显示等，在其他部分对于卡牌的判断、利用也都是以此类指针存储。

牌型类（`Play` 及其派生类）则用以区分不同的出牌方式和记录每次出牌，以及判断某一组牌是否符合当前牌型。

牌管理器类（`CardsManager`）以及卡牌显示控件（`CardBoard`）则用于记录和显示当前手牌和公示的上一轮出牌。

包管理器类（`PackageManager`）用于客户端之间传输中的拆包和封包，并同时处理可能出现的粘包等情况。

资源文件包括图标类、背景类以及卡牌类。

### 2. 项目自建类

#### 2.1. `class ClientWindow*`

A 客户端中利用成员函数 `initServer()` 开始接听其他客户端的连接请求，当 `newConnection` 达到两次时，新建 `GameWindow` 窗口开始搭建游戏界面以及网络通信架构（`start()` 函数）。

B、C 客户端则通过 `connectHost()` 函数连接指定主机（默认为 `QHostAddress::LocalHost`），在收到主机（A）的**成功连接信号**时新建 `GameWindow` 窗口开始搭建，并向 A 发出**连接完成的信号**。

其中 B、C 客户端还设置了菜单栏用以自行设置连接的主机 IP 地址。

#### 2.2. `class GameWindow`

本类中通过 `QHBoxLayout` 和 `QVBoxLayout` 布局安排各个控件（`QLabel`、`QPushButton`、`CardBoard`）的相对位置，在游戏进行到不同状态时，根据状态适当选择禁用/启用、隐藏/显示一些控件。

由于本项目仅有 AB 与 AC 的连接，在需要 BC 之间的通信时，会先由 B 传输给 A，进而由 A 传输给 C。

A 客户端同时作为服务器，会处理每次开局的随机状态信息，并分发给 B、C。

A 通过 `readPackageB()` 和 `readPackageC()` 两个函数接收来自 B 和 C 的信息，而 B 和 C 则通过 `readPackage()` 函数接收来自 A 的信息。

在包管理类和这些函数中，根据包头的不同会有不同的处理方式。

#### 2.3. `class Card`

本类通过以下成员数据来区分、比较、显示卡牌：

```cpp
private:
    bool isJoker; // 是否为大小王
    char type; // 花色或者大小王区分
    int num; // 数字，其中 A -> 1, J -> 11, Q -> 12, K -> 13
    bool chosen; // 是否被选择
    bool toBeChosen; // 是否要被选择（鼠标未放开）
    QImage *card = nullptr; // 卡牌对应的图片
```

实现了以下成员函数（包括运算符重载函数和构造函数）：

```cpp
public:
	explicit Card(char tp, QWidget *parent = nullptr); // 大小王构造函数
    explicit Card(char tp, int n, QWidget *parent = nullptr); // 其他牌构造函数
    bool operator<(const Card&) const; // 牌大小比较，包括花色
    bool operator>(const Card&) const; // 同上
    void chosed(); // 选择本牌
    QString getName(); // 获取本牌的名字（跟资源文件对应）
```

同时由于本类作为本项目的一个基础类，大多数类都会用到相关信息，因此大多数类都是其友元类。

#### 2.4. `class Play` 及其派生类

`Play` 类作为牌型基类，用于存储不同的牌型以及判断合法。

其含有虚成员函数：

```cpp
virtual bool isLegal(QVector<Card*>);
```

用于判断牌型合法性。

拥有静态成员：

```cpp
static Play *nullPlay;
```

表示无牌型要求（首回合或者其他两人均不出的情况）。

还有成员：

```cpp
QVector<Card*> cards;
```

用以存储某一个具体组合牌中的牌序列。

`Play` 作为基类还派生出了以下类：

```cpp
class Single; // 单牌
class Double; // 对子
template <unsigned n>
class Triple; // n 连三不带（无翅膀的飞机）
template <unsigned n>
class Straight; // 长度 n 顺子
template <unsigned n>
class DoubleStraight; // 长度 n 连对
class Bomb; // 普通炸弹
class JokerBomb; // 王炸
class QuartetPlusTwoSingle; // 四带二，带两张单牌
class QuartetPlusTwoDouble; // 四带二，带两个对子
template <unsigned n>
class TripleStraightPlusSingle; // n 连三带一（带翅膀的飞机）
template <unsigned n>
class TripleStraightPlusDouble; // n 连三带二（带翅膀的飞机）
```

每个类各自实现了牌型合理性判断函数和构造函数（用于将给定的牌排列成合理顺序）。

其中一些用模板实现的模板类只有头文件而没有实现文件，其他类则都具有头文件和实现文件。

#### 2.5. `class CardsManager`

主要用于存储当前手牌、公示牌的牌序列，有一个私有成员：

```cpp
QVector<Card*> cards;
```

头文件中包含了前面的各种牌型类的头文件，方便其他类的引用。

#### 2.6. `class CardBoard`

本类继承 `QWidget` 类，作为控件。

主要包含了以下成员：

```cpp
private:
    CardsManager *manager; // 牌管理器
    int pos = -1; // 鼠标点击按下的牌的位置
    int status = OTHERS; // 当前状态，OTHERS -> 0 表示其他玩家回合，SELF -> 1 表示自己回合，END -> 2 表示游戏结束
    int w, h; // 控件固定的长宽
```

并重写了以下虚函数：

```cpp
protected:
    void paintEvent(QPaintEvent*); // 用于将牌管理器中的卡牌较为紧凑的显示
private:
    void mousePressEvent(QMouseEvent*);
    void mouseReleaseEvent(QMouseEvent*);
    void mouseMoveEvent(QMouseEvent*);
	// 用于处理鼠标点击事件，通过这三个函数可以实现鼠标拖拽以单次选中（或反选）一个区间中的卡牌
```

#### 2.7. `class LandlordBoard`

类似上一个类，但用于显示地主牌，其中：

```cpp
void paintEvent(QPaintEvent*);
```

此函数会根据地主牌的空与不空（是否公开）来选择显示卡背还是具体开拍。

#### 2.8. `class PackageManager`

作为包管理工具类，用单例模式实现。

主要实现了两个功能：

```cpp
void sendPackage(GameWindow *gameWindow, QTcpSocket *writeSocket, ushort type);
int readPackage(GameWindow *gameWindow, QTcpSocket *readSocket, int &out);
```

分别用于将数据封包传输，以及将获取得的包拆包得到数据。

包分为包头和包身，根据不同类型的信息设计了不同的包头来区分，主要如下：

```
1: 0/1 选地主相关
2: 0/1 出牌相关
3: started 0/1/2 开始
4: restart 0/1/2 重新开始
5: 17~33 发牌
6: 34~50 发牌
7: 51~53 发地主牌
8: exit 0/1/2 -> A/B/C 退出相关
9: begin 0/1/2 -> A/B/C 随机选择叫地主
10: landloard cards 公示地主牌
```

### 3. 项目实现逻辑

#### 3.1. 客户端工作流程：窗口变换

主要包含的类见上。

在开始界面通过点击“开始连接”按钮调用 `initServer()` 函数或 `connectHost()` 函数来相互连接，并且客户端通过：

```cpp
connect(listenSocket, &QTcpServer::newConnection, this, &ClientWindowA::acceptConnection);
```

来通过 `acceptConnection()` 槽函数处理新连接，而客户端 B、C 则通过 `waitForConnected()` 函数来等待与 A 的连接。

在客户端 A 获得两个新连接时跳转到游戏界面，并向 B、C 发出开始信号，客户端 B、C 在接收到开始信号后跳转。

#### 3.2. 客户端工作流程：游戏界面中的状态

在游戏开始后，叫地主阶段和出牌阶段时，通过将每次做出的出牌或不出的选择向其他两个客户端广播，使得每个客户端的 `board->status` 和 `landlord` 得以更新。

客户端主要包含的状态如下：

```cpp
Play *req; // 上家出的手牌，如果上两家都不出则为 Play::nullPlay
int gaming; // 当前是哪个玩家的回合，A -> 0, B -> 1, C -> 2
int pre; // 上一个出牌的玩家，如果上两家都未出则为 -1
bool out; // 当前回合玩家是否出牌
int rest[3]; // 三个玩家的手牌剩余数量
int landlord; // 地主玩家，如果还未选出则为 -1
int randBegin; // 随机选择的开始叫地主的玩家
bool getLandlord; // 是否有人叫地主，根据此选择将后面的叫地主/不叫修改为抢地主/不抢
```

在每次客户端状态更新后（包括收到其他客户端信息或者本客户端做出选择），根据当前状态改变按钮文字提示（三组按钮用的是同一组控件），或者禁用/启用、隐藏/显示对应按钮，使得玩家只能在自己回合时得以操作。

并且还会根据当前更新的状态，显示/改变/隐藏一些文字提示，比如地主、农民身份，不出、不叫、不抢等文字信息。

在当前回合非自己回合（`boarder->status = OTHERS`）或不是出牌阶段时，禁用 `CardBoard` 类的鼠标事件，使得其无法操作手牌序列。

#### 3.3. 客户端工作流程：选择卡牌

在 `CardBoard` 类中，通过 `mousePressEvent()` 记录按下的坐标，仅处理按下位置为卡牌区域的事件。

`mouseMoveEvent()` 根据鼠标移动到的位置判断，如果移动到合法卡牌位置则将与按下的位置之间的所有卡牌加以阴影处理（通过 `toBeChosen` 状态和重绘）。

`mouseReleaseEvent()` 将当前 `toBeChosen` 状态为 `true` 的卡牌的 `chosen` 状态反选，并且清空 `toBeChosen` 状态，调用重绘来重新确定当前的选择状态。

在 `mouseReleaseEvent()` 中处理完当前选择状态后，还会根据当前选择卡牌的合理性（是否为合理牌型以及是否能压 `req` 中的牌）选择启用/禁用出牌按钮，也就是达到判断能否出牌的作用。

#### 3.4. 客户端工作流程：重新开始与退出

点击重新开始后，会向客户端 A 发出信号，A 中通过记录 `ifRestart` 判断是否三个客户端都选择重新开始。

如果同时选择重新开始，则由 A 向 B、C 发送重新开始信号。

如果一个客户端选择退出，则另外两个客户端收到信号会提示断开连接。

#### 3.5. 规则设计流程

同上面所说的牌型类，通过重载 `<` 和 `>` 运算符来判断同类牌大小（炸弹特判）。

其中多数牌型通过**存储最小的非带牌的大小来**实现，王炸则不考虑大小（压所有牌型）。

通过重写：

```cpp
bool isLegal(QVector<Card*>);
```

来判断给定牌集合是否符合某一种类型的牌。

具体到出牌时，根据出牌数量判断各种可能的牌型，所有可能如下：

```cpp
/*
 *  1: 1
 *  2: 2 or joker * 2
 *  3: 3
 *  4: 3 + 1 or 4
 *  5: 3 + 2 or 1 * 5
 *  6: 2 * 3 or 3 + 3 or 1 * 7 or 4 + 1 * 2
 *  7: 1 * 7
 *  8: 2 * 4 or (3 + 1) * 2 or 1 * 8 or 4 + 2 * 2
 *  9: 3 * 3 or 1 * 9
 * 10: 2 * 5 or (3 + 2) * 2 or 1 * 10
 * 11: 1 * 11
 * 12: 2 * 6 or 3 * 4 or (3 + 1) * 3 or 1 * 12
 * 13:
 * 14: 2 * 7
 * 15: 3 * 5 or (3 + 2) * 3
 * 16: 2 * 8 or (3 + 1) * 4
 * 17:
 * 18: 2 * 9 or 3 * 6
 * 19:
 * 20: 2 * 10 or (3 + 1) * 5 or (3 + 2) * 4
 */
```

#### 3.6. 通信协议

在 `PackageManager` 类中实现通信协议。

考虑到确实存在的粘包现象，即多次发出去的包可能只接受一次，所以需要设计包格式。

包分为包头和包身，其中包头占四个字节即两个 `ushort`，第一个表示包的类型（同上介绍），第二个表示包的整体长度。

包身中则根据不同类型填充不同长度的数据，比如传输卡牌信息时，通过统一长度写入卡牌名字。

在 `readPackage()` 拆包时，首先将 `readSocket` 中所有的未读出内容，与之前剩下的 `restBuffer` 合并。

如果当前合并得到 `restBuffer` 长度不够包头或者长度不够包头中包整体长度，则返回 $0$ 表示没有读到完整的包（没有获取数据）。

否则根据包头的类型读取其中的数据，并对当前状态进行修改。

在读取完后，再次判断剩余长度，如果不够包头或者不够包头中整体长度，则返回 $1$ 表示读取到一个包，并且之后没有完整的包。

如果长度足够，则返回 $2$ 表示读取完一个包后还有完整的包，提示调用的代码继续读取信息。

包的写入大致为：

```cpp
QByteArray sendByte;
QDataStream out(&sendByte, QIODevice::WriteOnly);
out.setByteOrder(QDataStream::BigEndian);
out << ushort(0) << ushort(0);
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
}

```

### 4. UI 设计

#### 4.1. 整体界面

![界面1](https://img.wzf2000.top/image/2020/09/23/103f3e3bae09f237b.png)

![界面2](https://img.wzf2000.top/image/2020/09/23/2de6ee6a42502c920.png)

![界面3](https://img.wzf2000.top/image/2020/09/23/3.png)

![界面4](https://img.wzf2000.top/image/2020/09/23/4.png)

#### 4.2. 控件样式

按钮样式为：

```css
QPushButton {
    border-radius: 15px;
	background-color: rgb(27, 154, 247);
    border-color: rgb(27, 154, 247);
    color: #FFF;
    font-size: 28px;
    height: 40px;
    line-height: 40px;
    padding: 0 30px;
	font-weight: 400;
	font-family: "华文楷书";
	text-decoration: none;
	text-align: center;
	margin: 5px;
}
QPushButton:visited {
    color: #FFF;
}
QPushButton:hover, QPushButton:focus {
    background-color: #4cb0f9;
    border-color: #4cb0f9;
    color: #FFF;
	text-decoration: none;
    outline: none;
}

QPushButton.disabled, QPushButton.is-disabled, QPushButton:disabled {
    top: 0 !important;
    background: #EEE !important;
    border: 1px solid #DDD !important;
    color: #CCC !important;
    opacity: .8 !important;
}

```

`QLabel` 控件通过设置 `QFont` 字体以及 HTML 填入来控制样式。

对于牌的展示控件，通过自己重写 `paintEvent()` 函数来使得牌的位置较为合理。

#### 4.3. 其他部分

为程序增加了图标，并且给开始界面和游戏界面添加了符合斗地主氛围的背景。

### 5. 拓展功能

#### 5.1. 卡牌多选

通过重写三个鼠标事件，实现了鼠标按住拖拽可以多选牌的功能。

#### 5.2. 自填主机地址

在客户端 B、C 的菜单栏增加了自己设置主机（A）IP 地址的功能。

![IP](https://img.wzf2000.top/image/2020/09/23/IP.png)

#### 5.3. 退出提示

在一个客户端退出时，会给其他两个客户端发出信号，另外两个客户端会跳出连接已断开的提示。

![退出](https://img.wzf2000.top/image/2020/09/23/fc73661d43743ac96f2235d159225070.png)

#### 5.4. 特殊牌型

本程序支持三个以上连的飞机，但对于类似 $333444555666$ 的飞机会判断为 $4$ 连三不带。

