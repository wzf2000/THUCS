echo "Start to compile..."
g++ AVL.cpp -o AVL -std=c++14
g++ RBT.cpp -o RBT -std=c++14
g++ splay.cpp -o splay -std=c++14
g++ rand_normal.cpp -o rand_normal -std=c++11
echo "Finished compiling!"
echo "Test 1:"
./rand_normal > normal_1.in
./AVL < normal_1.in > 1.out
./RBT < normal_1.in > 2.out
./splay < normal_1.in > 3.out
echo "Test 2:"
./rand_normal > normal_2.in
./AVL < normal_2.in > 1.out
./RBT < normal_2.in > 2.out
./splay < normal_2.in > 3.out
echo "Test 3:"
./rand_normal > normal_3.in
./AVL < normal_3.in > 1.out
./RBT < normal_3.in > 2.out
./splay < normal_3.in > 3.out
echo "Test 4:"
./rand_normal > normal_4.in
./AVL < normal_4.in > 1.out
./RBT < normal_4.in > 2.out
./splay < normal_4.in > 3.out
echo "Test 5:"
./rand_normal > normal_5.in
./AVL < normal_5.in > 1.out
./RBT < normal_5.in > 2.out
./splay < normal_5.in > 3.out
echo "Test 6:"
./rand_normal > normal_6.in
./AVL < normal_6.in > 1.out
./RBT < normal_6.in > 2.out
./splay < normal_6.in > 3.out