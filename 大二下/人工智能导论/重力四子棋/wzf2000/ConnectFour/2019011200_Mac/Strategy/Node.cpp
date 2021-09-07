//
//  Node.cpp
//  Strategy
//
//  Created by 王哲凡 on 4/17/21.
//  Copyright © 2021 Yongfeng Zhang. All rights reserved.
//

#include "Node.hpp"
#include "UCT.hpp"

Node **Node::nodePool = new Node*[MAX_SIZE];
size_t Node::cnt = 0;

Node::Node(int M, int N, int noX, int noY, int _x, int _y, bool turn, Node *par) :
    M(M), N(N), noX(noX), noY(noY), _x(_x), _y(_y), turn(turn), par(par),
    optionNum(0), options(new int[N]), child(new Node*[N]) {
    
    for (int i = 0; i < N; ++i) {
        if (UCT::top[i]) options[optionNum++] = i;
        child[i] = nullptr;
    }
    
}

Node::~Node() {
}

int Node::x() const {
    return _x;
}

int Node::y() const {
    return _y;
}

bool Node::getTurn() const {
    return turn;
}

bool Node::end() {
    if (flag != -1) return flag;
    if (_x == -1 && _y == -1) return flag = 0;
    if (!turn && machineWin(_x, _y, M, N, UCT::board)) return flag = 1;
    if (turn && userWin(_x, _y, M, N, UCT::board)) return flag = 1;
    if (isTie(N, UCT::top)) return flag = 1;
    return flag = 0;
}

bool Node::canExpand() const {
    return optionNum > 0;
}

Node *Node::expand() {
    bool newTurn = !turn;
    if (!expanded) {
        expanded = 1;
        int newY = -1;
        if ((newY = canWin(turn)) != -1) {
            int newX = --UCT::top[newY];
            UCT::board[newX][newY] = int(turn) + 1;
            if (noX == newX - 1 && noY == newY) --UCT::top[newY];
            optionNum = 0;
            return child[newY] = newNode(M, N, noX, noY, newX, newY, newTurn, this);
        }
        if ((newY = canWin(!turn)) != -1) {
            int newX = --UCT::top[newY];
            UCT::board[newX][newY] = int(turn) + 1;
            if (noX == newX - 1 && noY == newY) --UCT::top[newY];
            optionNum = 0;
            return child[newY] = newNode(M, N, noX, noY, newX, newY, newTurn, this);
        }
    }
    int newX, newY;
    while (1) {
        int option = rand() % optionNum;
        newY = options[option];
        std::swap(options[option], options[--optionNum]);
        newX = --UCT::top[newY];
        UCT::board[newX][newY] = int(turn) + 1;
        if (noX == newX - 1 && noY == newY) --UCT::top[newY];
        if (!UCT::top[newY] || !optionNum) break;
        UCT::board[UCT::top[newY] - 1][newY] = int(!turn) + 1;
        if (!turn && machineWin(UCT::top[newY] - 1, newY, M, N, UCT::board)
            || turn && userWin(UCT::top[newY] - 1, newY, M, N, UCT::board)) {
            UCT::board[UCT::top[newY] - 1][newY] = 0;
            if (noX == UCT::top[newY] && noY == newY) ++UCT::top[newY];
            UCT::board[newX][newY] = 0;
            ++UCT::top[newY];
        } else break;
    }
    return child[newY] = newNode(M, N, noX, noY, newX, newY, newTurn, this);
}

void Node::backPropagation(double change) {
    for (Node *node = this; node; node = node->par) {
        ++node->count;
        node->profit += change;
    }
}

Node *Node::bestChild(bool ifSet) {
    double maxScores = -1e20;
    int y = -1;
    Node *ret = nullptr;
    for (int i = 0; i < N; ++i) {
        if (!child[i]) continue;
        double trueChildProfit = (turn ? 1 : -1) * child[i]->profit;
        double scores = trueChildProfit / child[i]->count + c * sqrt(2 * log(count) / child[i]->count);
        // double scores = -child[i]->worstChild();
        if (scores > maxScores) {
            maxScores = scores;
            ret = child[i];
            y = i;
        }
    }
    if (ifSet) {
        int x = --UCT::top[y];
        UCT::board[x][y] = int(turn) + 1;
        if (x - 1 == noX && y == noY) --UCT::top[y];
    }
    return ret;
}

double Node::worstChild() {
    double maxScores = -1e20;
    for (int i = 0; i < N; ++i) {
        if (!child[i]) continue;
        double trueChildProfit = (turn ? 1 : -1) * child[i]->profit;
        double scores = trueChildProfit / child[i]->count + c * sqrt(2 * log(count) / child[i]->count);
        if (scores > maxScores) {
            maxScores = scores;
        }
    }
    return maxScores;
}

Node *Node::newNode(int M, int N, int noX, int noY, int _x, int _y, bool turn, Node *par) {
    if (cnt >= MAX_SIZE) {
        return new Node(M, N, noX, noY, _x, _y, turn, par);
    }
    if (!nodePool[cnt]) {
        return nodePool[cnt++] = new Node(M, N, noX, noY, _x, _y, turn, par);
    }
    nodePool[cnt]->set(M, N, noX, noY, _x, _y, turn, par);
    return nodePool[cnt++];
}

void Node::set(int M, int N, int noX, int noY, int _x, int _y, bool turn, Node *par) {
    this->M = M;
    this->N = N;
    this->noX = noX;
    this->noY = noY;
    this->_x = _x;
    this->_y = _y;
    this->turn = turn;
    this->par = par;
    optionNum = 0;
    count = 0;
    profit = 0;
    flag = -1;
    expanded = 0;
    for (int i = 0; i < N; ++i) {
        if (UCT::top[i]) options[optionNum++] = i;
        child[i] = nullptr;
    }
}

int Node::canWin(bool turn) {
    for (int i = 0; i < optionNum; ++i) {
        int y = options[i];
        int x = UCT::top[y] - 1;
        UCT::board[x][y] = int(turn) + 1;
        if (turn && machineWin(x, y, M, N, UCT::board)) {
            UCT::board[x][y] = 0;
            return y;
        }
        if (!turn && userWin(x, y, M, N, UCT::board)) {
            UCT::board[x][y] = 0;
            return y;
        }
        UCT::board[x][y] = 0;
    }
    return -1;
}
