//
//  Node.hpp
//  Strategy
//
//  Created by 王哲凡 on 4/17/21.
//  Copyright © 2021 Yongfeng Zhang. All rights reserved.
//

#ifndef Node_hpp
#define Node_hpp

#include <cstdio>
#include <cstdlib>
#include <cassert>
#include "Judge.h"

class Node {
    int M, N;
    int noX, noY;
    int _x, _y;
    bool turn;
    int optionNum, *options;
    Node *par, **child;
    double profit = 0;
    int count = 0;
    int flag = -1;
    bool expanded = 0;
    
public:
    Node(int M, int N, int noX, int noY, int _x = -1, int _y = -1, bool turn = 1, Node *par = nullptr);
    ~Node();
    int x() const;
    int y() const;
    bool getTurn() const;
    bool end();
    bool canExpand() const;
    Node *expand();
    void backPropagation(double change);
    Node *bestChild(bool ifSet);
    double worstChild();
    static Node *newNode(int M, int N, int noX, int noY, int _x = -1, int _y = -1, bool turn = 1, Node *par = nullptr);
    void set(int M, int N, int noX, int noY, int _x = -1, int _y = -1, bool turn = 1, Node *par = nullptr);
    int canWin(bool turn);

    static Node **nodePool;
    static size_t cnt;
};

const double c = 0.9;
const size_t MAX_SIZE = 10000000;

#endif /* Node_hpp */
