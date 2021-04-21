echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_add2.cpp -o rand_add2 -std=c++11
echo "Finished compiling!"
echo "Blocks = 1"
./rand_add2 1 > add2_1.in
./AVL < add2_1.in > 1.out
./RBT < add2_1.in > 2.out
./splay < add2_1.in > 3.out
echo "Blocks = 10"
./rand_add2 10 > add2_2.in
./AVL < add2_2.in > 1.out
./RBT < add2_2.in > 2.out
./splay < add2_2.in > 3.out
echo "Blocks = 100"
./rand_add2 100 > add2_3.in
./AVL < add2_3.in > 1.out
./RBT < add2_3.in > 2.out
./splay < add2_3.in > 3.out
echo "Blocks = 1000"
./rand_add2 1000 > add2_4.in
./AVL < add2_4.in > 1.out
./RBT < add2_4.in > 2.out
./splay < add2_4.in > 3.out
echo "Blocks = 10000"
./rand_add2 10000 > add2_5.in
./AVL < add2_5.in > 1.out
./RBT < add2_5.in > 2.out
./splay < add2_5.in > 3.out
echo "Blocks = 100000"
./rand_add2 100000 > add2_6.in
./AVL < add2_6.in > 1.out
./RBT < add2_6.in > 2.out
./splay < add2_6.in > 3.out
