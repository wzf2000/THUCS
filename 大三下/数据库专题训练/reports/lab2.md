## lab 2：日志管理 实验报告

### 一、基本功能

#### 1. 实现难点

##### 1.1. 添加记录WAL日志

通过 `LogManager` 相关接口来实际完成，主要难点在于在记录前确认新插入记录的 `RID` 等信息，需要配合 lab 1 中的逻辑完成（可考虑复用代码）。

##### 1.2. 完成三种基本操作的日志记录过程

通过调用 `LogFactory` 相关接口来构造 `Log` 对象，类比 `Begin()` 完成 `ATT` 表的更新，特别注意的是，需要根据 `DPT` 表中对应项是否存在来选择更新（不存在则更新）。

##### 1.3. 实现物理逻辑日志的序列化和反序列化

基本与 lab 1 中的序列化与反序列化思路相通，特别注意对于 `INSERT` 类型和 `DELETE` 类型分别不需要 `old_len_`、`old_val_` 和 `new_len_`、`new_val_` 的保存和恢复。

对于可变长度的字符串，需要先存储其长度，后存储数据。

##### 1.4. 完成 `Redo` 和 `Undo` 的具体操作

首先可以通过 `SystemManager` 和表名获取对应的表指针，进一步配合 `PageID` 又可以得到记录所在的页面。

特别地，在 `Redo` 前，需要保证记录的 `LSN` 大于页面的 `LSN`，否则可能导致重复的 `Redo`。

而具体 `Redo` 和 `Undo` 时，只需要根据 `UpdateLog` 的类型调用新增加的 `PageHandle::InsertRecord()` 等接口即可，`Undo` 时注意插入与删除逻辑相反。

##### 1.5. `Checkpoint` 日志的设计

基本与前面的序列化与反序列化一致，基本功能只需要保存 `ATT` 与 `DPT` 表即可。

##### 1.6. 完成故障恢复算法

`Redo` 时，从 `DPT` 表中记录的最小的 `LSN` 开始，到 `current_lsn_` 截止，对于 `UpdateLog` 类型的日志进行实际上的 `Redo`，并且注意与 1.2 中类似地更新 `DPT` 表。

特别注意，当遇到 `CheckpointLog` 类型时，之后的 `LSN` 便无须再 `Redo`（考虑到基本功能不需要考虑多个 `Checkpoint`）。

`Undo` 时，根据 `DPT` 表中的每一条记录，向前找 `PrevLSN`，对于 `UpdateLog` 进行 `Undo`，直到遇到 `Begin` 类型的日志为止。

#### 2. 实现耗时

基本功能总计耗时 $5$ 小时左右。

包含高级功能，总耗时约 $10$ 小时。

### 二、高级功能

#### 1. `Undo` 过程中系统出现异常的恢复

##### 1.1. 实现逻辑

按照课上要求实现了 `CLRLog` 的添加，修改了 `Undo` 过程，在 `Undo` 后会进行 `CLRLog` 的插入。

`CLRLog` 的记录与 `UpdateLog` 内容相似，但多了一个 `undoNext` 字段，用于表示 `undo` 时跳转的 `LSN`，数值上即为对应的 `UpdateLog` 的 `prevLSN`，除此之外，`CLRLog` 的新旧数据应该与 `UpdateLog` 相反。

分支名为 `lab2-advanced`，`commit id` 为 `4f83e06cf2ed8c4ce59463ec788731a2cdd04e08`。

##### 1.2. 测试验证

为了方便测试 `Undo` 过程中的系统出现的异常，我增加了一条指令：

```sql
usecrash <database>;
```

表示在 `use <database>;` 的过程中，随机在 `Undo` 的某个阶段发生 `crash;` 导致 `use` 命令失败返回 `FAILURE`。

在原有基本测例的基础上，我编写了以下测例 `50_undo_crash.sql`：

```sql
use dbtrain_test_lab2;

begin;

delete from persons where id > 190;

commit;

begin;

insert into persons values(203, '12345678901234567890', '09876543210987654321', 36.3);

insert into persons values(204, '12345678901234567890', '09876543210987654321', 36.3);

insert into persons values(205, '12345678901234567890', '09876543210987654321', 36.3);

insert into persons values(206, '12345678901234567890', '09876543210987654321', 36.3);

checkpoint;

crash;

usecrash dbtrain_test_lab2;

use dbtrain_test_lab2;

select * from persons;

```

期望结果即 `usecrash dbtrain_test_lab2;` 相当于未发生：

```
-- 1.use dbtrain_test_lab2;
SUCCESS

-- 2.begin;
SUCCESS

-- 3.delete from persons where id > 190;
SUCCESS

-- 4.commit;
SUCCESS

-- 5.begin;
SUCCESS

-- 6.insert into persons values(203, '12345678901234567890', '09876543210987654321', 36.3);
SUCCESS

-- 7.insert into persons values(204, '12345678901234567890', '09876543210987654321', 36.3);
SUCCESS

-- 8.insert into persons values(205, '12345678901234567890', '09876543210987654321', 36.3);
SUCCESS

-- 9.insert into persons values(206, '12345678901234567890', '09876543210987654321', 36.3);
SUCCESS

-- 10.checkpoint;
SUCCESS

-- 11.crash;
CRASH

-- 12.usecrash dbtrain_test_lab2;
FAILURE

-- 13.use dbtrain_test_lab2;
SUCCESS

-- 14.select * from persons;
id | first_name | last_name | temperature
1 | 12345678901234567890 | 09876543210987654321 | 36.3
...

Bye

```

经过多次随机验证（随机种子不同），均通过测试，我认为实现完成了目标。

