## lab 4：优化器 实验报告

### 一、基本功能

#### 1. 实现难点

##### 1.1. 补全直方图的构建和估计函数

构建只需要按照 `num_buckets_` 平均分桶即可。

估计时，先计算完整的桶所占的实际比例并求和，再加上根据所在的部分桶的大小比例乘以所在桶的总比例即可。

##### 1.2. 补全读取表的数据并构建直方图的过程

构建 `TableScanNode` 节点，利用 `Next()` 成员函数获取所有的记录，传入调用 1.1 实现的直方图的 `Init()` 函数即可。

##### 1.3. 补全 `Filter` 算子的基数估计过程

首先判断 `cond_` 为 `nullptr` 的情况，即直接返回孩子的 `Cost()` 结果。

其次尝试通过 `dynamic_cast` 将 `cond_` 转为 `AlgebraCondition*` 类型，如果成功则利用 `UpdateBound()` 获取上下界，调用 1.1 中实现的对应估计函数即可，注意需要将结果乘上孩子的 `Cost()` 结果再返回。

最后 `cond_` 一定为 `AndCondition*` 类型，通过队列保存所有的条件，将其中 `AlgebraCondition*` 类型更新对应列的上下界，`AndCondition*` 类型则将其所有 `conds` 中的元素加入队列，循环至结束，再对各列分别利用直方图统计函数计算比例，最后相乘并乘上孩子的 `Cost()` 结果即可。

##### 1.4. 完成 `Optimizer` 中连接顺序优化过程

首先利用已知的连接关系边建图，统计所有表的 `Cost()` 并记录，找到最小的表，并加入 `priority_queue` 中维护。

之后只需要利用 `priority_queue` 来执行类似 Prim 算法的过程即可，即每次取出堆顶，并将其相邻的其他表（且尚未加入连接树）依次加入 `priority_queue`，按照取出的顺序添加与之相连尚未处理的连接关系即可。

#### 2. 实现耗时

基本功能总计耗时 $3$ 小时左右。

包含高级功能，总耗时约 $15$ 小时。

### 二、高级功能

#### 0. 实现代码

我的实现所在分支名为 `lab4-advanced`，`commit id` 为 `f3172d35d05d8b2a01a99a0d1a17eacdd6dea9fa`，由于算法的更改，测试结果会与基础测例的标准不同，因此 ci 无法通过。

#### 1. Join 基数估计以及对应的优化

##### 1.1. 实现逻辑

我选择了 PostgreSQL 进行尝试实现，在查询文档、确认源码后，我得知 PostgreSQL 采用了将连接关系作为图，并将其看作旅行商问题（TSP，traveling salesman problem）的方式进行优化。

具体来说，我首先实现了 `LinearCounting` 基数统计：

- 当元素数量不超过 $500$ 时，我直接计算了不同元素的数量作为基数。
- 当元素数量超过 $500$ 时，我使用了 `hash` 和大小为 $m$ 比特的 `Bitmap` 统计了哈希桶中 $0$ 的个数 $\mu$，将 $-m \ln \dfrac{\mu}{m}$ 作为估计的基数。

然后我实现了连接基数估计（采取了课上所讲的方式）：
$$
T(R(X, Y) ⋈ S(Y, Z)) = \frac{T(R) \times T(S)}{\max(V(R, Y), V(S, Y))}
$$
此处需要提前设置 `JoinNode` 左右节点的表名，并计算左右节点的 `Cost()`。

最后，我修改了 `Optimizer` 中的连接顺序优化过程，将其改为了 Genetic Query Optimization（GEQO），即使用 Genetic-Algorithm（遗传算法）优化。

具体设置如下：

- 排除表数小于 $3$ 的情况，此时可以确定只有唯一连接方案。
- 适应度（`fitness`）的计算中，按基因组顺序加入节点，分别按照从前到后的顺序，尝试与前面节点构成的连通块进行合并（连接），配合并查集记录连通块的基数，计算连接中的产生的代价，并调用 `JoinNode` 的 `Cost()` 函数更新新连通块的基数。
  - 考虑到此处我的实现为 Nested Loop Join，所以我统一计算代价为两表基数之积。
- 对于交叉，我采用了比较简单的 order crossover，即选择一段使用父亲基因，其余按照顺序填充母亲基因。
  - 由于 ERX 和 PRX 算法都较复杂，且构造测例本身较简单，因此我未选择使用。
- 设置染色体池大小 `pool_size` 为 $2^{n + 1}$，其中 $n$ 为表数（并限制在 $[50, 250]$ 这个区间内），迭代代数与此相同。
- 通过 `random_shuffle()` 随机生成初始染色体池，并在计算适应度后按照从小到大排列。
- 每一轮迭代中：
  - 利用 `linear_rand()` 来获取随机的父母染色体，其按照分配了按照从前到后、从大到小线性变化的概率。
  - 将父母染色体交叉后得到子染色体，计算其适应度。
  - 替换原染色体群中的最后一个（适应度最低的），并执行插入排序保持有序性。
- 最后选择染色体群中适应度最高的，解析其连接顺序用于实际连接。

##### 1.2. 基础测例验证

对于基础样例 `30_multi_reorder.sql` 的第三条询问：

```sql
explain select t2.id from t1, t2, t3, t4 where t3.score < 80.0 and t1.id = t2.id and t1.id = t3.id and t1.id = t4.id;
```

首先可以计算得到各表的实际基数（过滤后）为 $10, 50, 74, 100$。

原标准输出为：

```
Select:
	Project Node:
		Join Node:
			Join Node:
				Join Node:
					Table Scan Node(t1):
					Table Scan Node(t2):
				Filter Node:
					Table Scan Node(t3):
			Table Scan Node(t4):
```

其实际损耗为（`t1` 与 `t3` 连接结果基数为 $7$）：
$$
10 \times 50 + 10 \times 74 + 7 \times 100 = 1940
$$
我设计的算法给出以下答案：

```
Select:
        Project Node:
                Join Node:
                        Join Node:
                                Table Scan Node(t2):
                                Join Node:
                                        Filter Node:
                                                Table Scan Node(t3):
                                        Table Scan Node(t1):
                        Table Scan Node(t4):
```

其实际损耗为：
$$
10 \times 74 + 7 \times 50 + 7 \times 100 = 1790
$$
在 Nested Loop Join 下确实优于原答案。

##### 1.3. 额外测例验证

我还设计了额外测例，可见分支下 `join-cost-test.py` 文件，其中各表中 `id2` 均为 $[1, 100]$ 的均匀分布（可基本认为各表的 distinct value 为 $100$）：

```python
import random

print('drop database if exists dbtrain_test_lab4;');
print('create database dbtrain_test_lab4;')
print('use dbtrain_test_lab4;')

def build(tname, n, m):
    print('')
    print(f'create table {tname}(id int, id2 int);')
    print('')
    values = []
    for i in range(1, n + 1):
        values.append(f'({i}, {random.randint(1, m)})')

    for j in range(0, n, 100):
        merge_value = ', '.join(values[j : j + 100])
        print(f'insert into {tname} values {merge_value};')
        print('')

build('t5', 600, 100)
build('t6', 1200, 100)
build('t7', 600, 100)
build('t8', 1200, 100)
build('t9', 600, 100)
build('t10', 1200, 100)

print('\nanalyze;')
print('\nexplain select t5.id from t5, t6, t7, t8, t9, t10 where t5.id < 301 and t7.id < 101 and t9.id < 201 and t6.id < 1001 and t10.id < 901 and t5.id2 = t6.id2 and t6.id2 = t7.id2 and t7.id2 = t8.id2 and t8.id2 = t9.id2 and t9.id2 = t10.id2;')
print('\nexplain select t5.id from t5, t6, t7, t8 where t5.id < 301 and t7.id < 101 and t8.id < 1001 and t5.id2 = t6.id2 and t6.id2 = t7.id2 and t7.id2 = t8.id2;')

```

通过 `python join-cost-test.py | ./build/bin/main` 可获取结果。

其中包含两个询问：

```sql
explain select t5.id from t5, t6, t7, t8, t9, t10 where t5.id < 301 and t7.id < 101 and t9.id < 201 and t6.id < 1001 and t10.id < 901 and t5.id2 = t6.id2 and t6.id2 = t7.id2 and t7.id2 = t8.id2 and t8.id2 = t9.id2 and t9.id2 = t10.id2;
explain select t5.id from t5, t6, t7, t8 where t5.id < 301 and t7.id < 101 and t8.id < 1001 and t5.id2 = t6.id2 and t6.id2 = t7.id2 and t7.id2 = t8.id2;
```

两次 `explain` 的结果如下：

```
Select:
        Project Node:
                Join Node:
                        Join Node:
                                Join Node:
                                        Table Scan Node(t8):
                                        Filter Node:
                                                Table Scan Node(t7):
                                Join Node:
                                        Filter Node:
                                                Table Scan Node(t10):
                                        Filter Node:
                                                Table Scan Node(t9):
                        Join Node:
                                Filter Node:
                                        Table Scan Node(t5):
                                Filter Node:
                                        Table Scan Node(t6):

Select:
        Project Node:
                Join Node:
                        Join Node:
                                Join Node:
                                        Filter Node:
                                                Table Scan Node(t7):
                                        Table Scan Node(t6):
                                Filter Node:
                                        Table Scan Node(t5):
                        Filter Node:
                                Table Scan Node(t8):
```

以第一次询问为例，六张表的实际基数（过滤后）分别为 $300, 1000, 100, 1200, 200, 900$，按照基础要求连接顺序为 `t7-t6-t5-t8-t9-t10`，对应的实际损耗可估计为：
$$
100 \times 1000 + 1000 \times 300 + 3000 \times 1200 + 36000 \times 200 + 72000 \times 900 = 76 000 000
$$
而我设计的连接顺序为 `((t7-t8)-(t10-t9))-(t5-t6)`，对应的实际损耗可估计为：
$$
100 \times 1200 + 900 \times 200 + 300 \times 1000 + 1200 \times 1800 + 21600 \times 3000 = 67 560 000
$$
可以看到同样损耗更小。

