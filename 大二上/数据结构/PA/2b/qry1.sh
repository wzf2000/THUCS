echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_qry1.cpp -o rand_qry1 -std=c++11
echo "Finished compiling!"
echo "Adds = 50000"
echo "Test 1:"
./rand_qry1 50000 > qry1_1.in
./AVL < qry1_1.in > 1.out
./RBT < qry1_1.in > 2.out
./splay < qry1_1.in > 3.out
echo "Test 2:"
./rand_qry1 50000 > qry1_1.in
./AVL < qry1_1.in > 1.out
./RBT < qry1_1.in > 2.out
./splay < qry1_1.in > 3.out
echo "Test 3:"
./rand_qry1 50000 > qry1_1.in
./AVL < qry1_1.in > 1.out
./RBT < qry1_1.in > 2.out
./splay < qry1_1.in > 3.out
echo "Test 4:"
./rand_qry1 50000 > qry1_1.in
./AVL < qry1_1.in > 1.out
./RBT < qry1_1.in > 2.out
./splay < qry1_1.in > 3.out
echo "Adds = 300000"
echo "Test 1:"
./rand_qry1 300000 > qry1_2.in
./AVL < qry1_2.in > 1.out
./RBT < qry1_2.in > 2.out
./splay < qry1_2.in > 3.out
echo "Test 2:"
./rand_qry1 300000 > qry1_2.in
./AVL < qry1_2.in > 1.out
./RBT < qry1_2.in > 2.out
./splay < qry1_2.in > 3.out
echo "Test 3:"
./rand_qry1 300000 > qry1_2.in
./AVL < qry1_2.in > 1.out
./RBT < qry1_2.in > 2.out
./splay < qry1_2.in > 3.out
echo "Test 4:"
./rand_qry1 300000 > qry1_2.in
./AVL < qry1_2.in > 1.out
./RBT < qry1_2.in > 2.out
./splay < qry1_2.in > 3.out
echo "Adds = 500000"
echo "Test 1:"
./rand_qry1 500000 > qry1_3.in
./AVL < qry1_3.in > 1.out
./RBT < qry1_3.in > 2.out
./splay < qry1_3.in > 3.out
echo "Test 2:"
./rand_qry1 500000 > qry1_3.in
./AVL < qry1_3.in > 1.out
./RBT < qry1_3.in > 2.out
./splay < qry1_3.in > 3.out
echo "Test 3:"
./rand_qry1 500000 > qry1_3.in
./AVL < qry1_3.in > 1.out
./RBT < qry1_3.in > 2.out
./splay < qry1_3.in > 3.out
echo "Test 4:"
./rand_qry1 500000 > qry1_3.in
./AVL < qry1_3.in > 1.out
./RBT < qry1_3.in > 2.out
./splay < qry1_3.in > 3.out