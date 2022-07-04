## lab 3：执行器 实验报告

### 一、基本功能

#### 1. 实现难点

##### 1.1. 实现 `AndConditionNode` 和 `OrConditionNode` 的 `visit()` 函数

两种节点结果的表名可通过 `rhs_->accept(this)` 的返回值得到，需要注意将原有的 `table_filter_` 内的 `Condition` 加入到对应新 `Condition` 的 `conds` 中。

注意为了可嵌套使用，返回值也要与其他 `CondtionNode` 类似。

##### 1.2. 实现 `JoinConditionNode` 的 `visit()` 函数

与 1.1 基本类似。

##### 1.3. 在 `Select` 的 `visit()` 函数中添加连接算子的部分

对于两个表连接的维护，首先需要根据两表的 `table_shift_` 修改 `JoinCondition` 信息，其次新的 `JoinNode` 应该从两表 `find()` 得到的根节点构建。

并且在构建后，需要将合并中的右侧子树所有结点的 `table_shift_` 加上左侧子树总列数，以保持 `offset` 的合理性。

##### 1.4. 实现 `Next()` 函数，进行连接运算

先通过不断对两个儿子的 `Next()` 获取到完整的 `RecordList`，再进行简单的 Nested Loop Join 以确认每对 `Record` 是否满足 `cond_`。

#### 2. 实现耗时

基本功能总计耗时 $3$ 小时左右。

包含高级功能，总耗时约 $10$ 小时。

### 二、高级功能

#### 0. 实现代码

我的实现所在分支名为 `lab3-advanced`，`commit id` 为 `412241b2abc08c0ba5c681712f0e8ca96b1ef384`。

#### 1. 实现多种 join 算法

##### 1.1. 实现逻辑

我额外实现了 Hash Join 与 Sort Merge Join。

其中 Sort Merge，我采用先排序，后归并，再根据原有的额外编号重排序的方法得到结果列表。

对于 Hash Join，为了方便，我统一将右侧的表计算哈希值，放入哈希表中维护，枚举左侧的记录，计算哈希值在哈希表中进行精确匹配得到结果列表。

##### 1.2. 测试验证

我在原有测例基础上设计了 `large1, large2` 两张大表，大小均为 `4000`：

```sql
create table large1(id int, first_name varchar(20), last_name varchar(20), temperature float);
-- insert
create table large2(id int, temperature float);
-- insert
```

并通过类似原有测例的方式测试：

```sql
select * from large1, large2 where large1.id = large2.id;
```

经过测试，Nested Loop Join 的表现如下：

![image-20220424164028334](https://img.wzf2000.top/image/2022/04/24/image-20220424164028334.png)

Sort Merge Join 的表现如下：

![image-20220424164102096](https://img.wzf2000.top/image/2022/04/24/image-20220424164102096.png)

Hash Join 的表现如下：

![image-20220424164121077](https://img.wzf2000.top/image/2022/04/24/image-20220424164121077.png)

可以看到，整体速度上，后两者远快于第一种朴素方法（时间几乎不到 $\dfrac{1}{10}$），而 Hash Join 又略快于 Sort Merge Join。

这是因为 Nested Loop Join 的时间复杂度为 $O(nm)$，而在匹配数量不多的情况下，Sort Merge Join 的时间复杂度为 $O(n\log n + m\log m)$，Hash Join 的时间复杂度接近 $O(n + m)$，在 $n, m$ 不是那么大的情况下，后两者的实际效率接近。

#### 2. 实现聚合算子

##### 2.1. 实现逻辑

我通过修改了 `ProjectNode` 的方式来完成，其中对于存在 `SUM` 等算子列的情况，将结果聚合为一行。

对于算子的解析，我修改了 `sql.y` 等解析文件，添加了 `SumCol` 等类以及 `OpType` 来表示聚合类别情况，其中 `NONE` 表示普通列。

##### 2.2. 测试验证

我在原有测例基础上设计了以下测例用以验证：

```sql
use dbtrain_test_lab3;

select MAX(id), MIN(id), COUNT(*) from person1;

select AVG(id), SUM(temperature), COUNT(first_name) from person1 where id > 900;

select SUM(id), AVG(temperature) from person2 where id < 100 and id > 50 and temperature = 36.3;

select SUM(id), COUNT(*) from person1 where id > 1000;

```

经测试，可以得到结果：

```
-- 1.use dbtrain_test_lab3;
SUCCESS

-- 2.select MAX(id), MIN(id), COUNT(*) from person1;
MAX(id) | MIN(id) | COUNT(*)
1000 | 1 | 1000

-- 3.select AVG(id), SUM(temperature), COUNT(first_name) from person1 where id > 900;
AVG(id) | SUM(temperature) | COUNT(first_name)
950 | 3630 | 100

-- 4.select SUM(id), AVG(temperature) from person2 where id < 100 and id > 50 and temperature = 36.3;
SUM(id) | AVG(temperature)
3675 | 36.3

-- 5.select SUM(id), COUNT(*) from person1 where id > 1000;
SUM(id) | COUNT(*)
0 | 0

Bye

```

因此，我认为我的实现可以认为基本正确。

