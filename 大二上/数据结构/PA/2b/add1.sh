echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_add1.cpp -o rand_add1 -std=c++11
echo "Finished compiling!"
echo "Blocks = 1"
./rand_add1 1 > add1_1.in
./AVL < add1_1.in > 1.out
./RBT < add1_1.in > 2.out
./splay < add1_1.in > 3.out
echo "Blocks = 10"
./rand_add1 10 > add1_2.in
./AVL < add1_2.in > 1.out
./RBT < add1_2.in > 2.out
./splay < add1_2.in > 3.out
echo "Blocks = 100"
./rand_add1 100 > add1_3.in
./AVL < add1_3.in > 1.out
./RBT < add1_3.in > 2.out
./splay < add1_3.in > 3.out
echo "Blocks = 1000"
./rand_add1 1000 > add1_4.in
./AVL < add1_4.in > 1.out
./RBT < add1_4.in > 2.out
./splay < add1_4.in > 3.out
echo "Blocks = 10000"
./rand_add1 10000 > add1_5.in
./AVL < add1_5.in > 1.out
./RBT < add1_5.in > 2.out
./splay < add1_5.in > 3.out
echo "Blocks = 100000"
./rand_add1 100000 > add1_6.in
./AVL < add1_6.in > 1.out
./RBT < add1_6.in > 2.out
./splay < add1_6.in > 3.out
