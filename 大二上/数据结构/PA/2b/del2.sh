echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_del2.cpp -o rand_del2 -std=c++11
echo "Finished compiling!"
echo "Test 1:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out
echo "Test 2:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out
echo "Test 3:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out
echo "Test 4:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out
echo "Test 5:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out
echo "Test 6:"
./rand_del2 > del2.in
./AVL < del2.in > 1.out
./RBT < del2.in > 2.out
./splay < del2.in > 3.out