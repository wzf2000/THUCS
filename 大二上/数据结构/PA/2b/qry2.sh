echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_qry2.cpp -o rand_qry2 -std=c++11
echo "Finished compiling!"
echo "Block Size = 1"
./rand_qry2 1 > qry2_0.in
./AVL < qry2_0.in > 1.out
./RBT < qry2_0.in > 2.out
./splay < qry2_0.in > 3.out
echo "Block Size = 10"
./rand_qry2 10 > qry2_1.in
./AVL < qry2_1.in > 1.out
./RBT < qry2_1.in > 2.out
./splay < qry2_1.in > 3.out
echo "Block Size = 100"
./rand_qry2 100 > qry2_2.in
./AVL < qry2_2.in > 1.out
./RBT < qry2_2.in > 2.out
./splay < qry2_2.in > 3.out
echo "Block Size = 1000"
./rand_qry2 1000 > qry2_3.in
./AVL < qry2_3.in > 1.out
./RBT < qry2_3.in > 2.out
./splay < qry2_3.in > 3.out
echo "Block Size = 10000"
./rand_qry2 10000 > qry2_4.in
./AVL < qry2_4.in > 1.out
./RBT < qry2_4.in > 2.out
./splay < qry2_4.in > 3.out
echo "Block Size = 100000"
./rand_qry2 100000 > qry2_5.in
./AVL < qry2_5.in > 1.out
./RBT < qry2_5.in > 2.out
./splay < qry2_5.in > 3.out
