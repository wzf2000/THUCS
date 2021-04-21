#ifndef RAND_H
#define RAND_H

#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <chrono>
#include <random>

const int MAXN = 1000000;
const int MAXX = 8500000;

unsigned t;

void init_rand()
{
    t = std::chrono::system_clock::now ().time_since_epoch().count();
    srand(t);
}

template <class RandomAccessIterator>
void my_shuffle(RandomAccessIterator first, RandomAccessIterator last)
{
    std::shuffle(first, last, std::default_random_engine(t));
}

int get_rand(int l, int r)
{
    return rand() % (r - l) + l;
}

#endif
