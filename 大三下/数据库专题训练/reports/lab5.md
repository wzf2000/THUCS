## lab 5：并发控制 实验报告

### 一、基本功能

#### 1. 实现难点

##### 1.1. 添加操作隐藏列的接口

与 `GetRid()` 和 `SetRid()` 接口基本一致，分别添加两个隐藏列 `CreateXID` 和 `DeleteXID` 修改和获取的接口即可。

##### 1.2. 实现多线程场景下的记录页面的插入、删除和查找

`InsertRecord()` 中注意添加 `CreateXID` 隐藏列即可。

`DeleteRecord()` 中需要修改记录的 `DeleteXID` 再写回，而不进行位图的修改。

`LoadRecords()` 中需要筛选出不满足条件的数据进行过滤：

- 已提交的删除，即 `DeleteXID` 字段存在，且值小于当前的事务 ID，并且其不在 `uncommit_xids` 中。
- 未提交的插入，即 `CreateXID` 字段的值大于当前的事务 ID 或其在 `uncommit_xids` 中。

##### 1.3. 修改多线程场景下的表的插入、删除和更新

`InsertRecord()` 和 `DeleteRecord()` 只需要换用对应的新接口即可。

`UpdateRecord()` 则先调用 `DeleteRecord()` 再调用 `InsertRecord()` 即可，同时需要修改对应的 `meta_` 信息。

##### 1.4. `CheckPoint` 恢复当前事务编号

在 `Load()` 和 `Store()` 时额外存储读取 `TxManager` 的 `current_xid_` 即可。

#### 2. 实现耗时

基本功能总计耗时 $5$ 小时左右。

包含高级功能，总耗时约 $10$ 小时。

### 二、高级功能

#### 0. 实现代码

我的实现所在分支名为 `lab5-advanced`，`commit id` 为 `805880b10c15ede258e4e9f1128ef28a35d4064d`。

#### 1. MVCC 的垃圾回收

##### 1.1. 实现逻辑

我选择在关闭数据库时进行垃圾回收，也即 `SystemManager::CloseDatabase()` 被调用的时候进行 `Collect()`。

回收时，扫描数据库中的所有表，获取其中的每一条记录（类似 `LoadRecords()`），对于其中存在合法 `DeleteXID` 字段的记录，说明记录已经被删除，调用原来的 `DeleteRecord` 接口将其删除。

特别考虑到，原有表 `meta_` 信息中的 `first_free_` 字段的维护可能导致，垃圾回收只有第一张表有效（后续仍然选择开辟新表），我修改了获取 `GetNextFree()` 的逻辑，改为了顺序扫描页号，手动确认是否为空，仅在所有页面确实满载的情况下才选择新建页面。

##### 1.2. 额外测例验证

我设计了额外测例，可见分支下 `test_out.py` 文件：

```python
import random

m = 20

def add(n, f):
    f.write('insert into persons_large values ')
    for i in range(n):
        if i == n - 1:
            f.write(f"({i}, '{chr(random.randint(0, 25) + ord('a'))}', 36.2);\n\n")
        else:
            f.write(f"({i}, '{chr(random.randint(0, 25) + ord('a'))}', 36.2), ")
    f.write('delete from persons_large where temperature = 36.2;\n')

def add_res(f, cnt):
    f.write(f'-- {cnt}.insert;\nSUCCESS\n\n')
    cnt += 1
    f.write(f"-- {cnt}.delete from persons_large where temperature = 36.2;\nSUCCESS\n\n")
    cnt += 1
    f.write('Bye\n')

with open('../dbtrain-lab-test/lab5/test/400_large_delete_build.sql', 'w+') as f:
    f.write('use dbtrain_test_lab5;\n\n')
    f.write('create table persons_large(id int, name varchar(20), temperature float);\n\n')
    add(100, f)

for _ in range(m):
    name = _ + 1
    with open(f'../dbtrain-lab-test/lab5/test/4{name:02}_large_delete.sql', 'w+') as f:
        f.write('use dbtrain_test_lab5;\n\n')
        add(90, f)
    
with open('../dbtrain-lab-test/lab5/result/400_large_delete_build.result', 'w+') as f:
    f.write('-- 1.use dbtrain_test_lab5;\nSUCCESS\n\n')
    f.write('-- 2.create table persons_large(id int, name varchar(20), temperature float);\nSUCCESS\n\n')
    add_res(f, 3)

for _ in range(m):
    name = _ + 1
    with open(f'../dbtrain-lab-test/lab5/result/4{name:02}_large_delete.result', 'w+') as f:
        f.write('-- 1.use dbtrain_test_lab5;\nSUCCESS\n\n')
        add_res(f, 2)
```

其中逻辑为，不断地进行一张表中记录的添加，并统一删除。

对于原 MVCC 场景，由于记录不会被真正删除，会导致最终表数据文件大小不断变大，实际测试如下：

![image-20220522154642272](https://img.wzf2000.top/image/2022/05/22/image-20220522154642272.png)

可以看到 `persons_large.data` 达到了 $92\,\mathrm{KB}$ 的大小也就是 $23$ 个物理页面。

而改进后：

![image-20220522154729501](https://img.wzf2000.top/image/2022/05/22/image-20220522154729501.png)

可以看到 `persons_large.data` 仅占到了 $8\,\mathrm{KB}$ 的大小，也即两个页面，可以认为垃圾回收成功。

