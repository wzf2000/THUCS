//
//  UCT.hpp
//  Strategy
//
//  Created by 王哲凡 on 4/17/21.
//  Copyright © 2021 Yongfeng Zhang. All rights reserved.
//

#ifndef UCT_hpp
#define UCT_hpp

#include <cstdio>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include "Node.hpp"

class UCT {
    Node *root = nullptr;
    int M, N;
    int noX, noY;
    
public:
    UCT(int M, int N, int noX, int noY);
    ~UCT();
    std::pair<int, int> search(int **board, const int *top, double startTime);
    Node *treePolicy(Node *node);
    double defaultPolicy(Node *node);
    int calcProfit(int **board, int *top, bool turn, int x, int y);
    void randPlay(int **board, int *top, bool turn, int &x, int &y);
    int willLose(bool turn);
    static double my_clock();


    static int **board, *top;
    
};

const double timeLimit = 2.7e6;

#endif /* UCT_hpp */
