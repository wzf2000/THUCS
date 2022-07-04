## 实验一：Cache 测量实验

<span style="float:right">计93 王哲凡 2019011200</span>

### 0. 实验环境 Cache 参数

使用 Coreinfo 指令查看得到如下设置：

```
Logical Processor to Cache Map:
**----------  Data Cache          0, Level 1,   32 KB, Assoc   8, LineSize  64
**----------  Instruction Cache   0, Level 1,   32 KB, Assoc   8, LineSize  64
**----------  Unified Cache       0, Level 2,  256 KB, Assoc   4, LineSize  64
************  Unified Cache       1, Level 3,   12 MB, Assoc  16, LineSize  64
--**--------  Data Cache          1, Level 1,   32 KB, Assoc   8, LineSize  64
--**--------  Instruction Cache   1, Level 1,   32 KB, Assoc   8, LineSize  64
--**--------  Unified Cache       2, Level 2,  256 KB, Assoc   4, LineSize  64
----**------  Data Cache          2, Level 1,   32 KB, Assoc   8, LineSize  64
----**------  Instruction Cache   2, Level 1,   32 KB, Assoc   8, LineSize  64
----**------  Unified Cache       3, Level 2,  256 KB, Assoc   4, LineSize  64
------**----  Data Cache          3, Level 1,   32 KB, Assoc   8, LineSize  64
------**----  Instruction Cache   3, Level 1,   32 KB, Assoc   8, LineSize  64
------**----  Unified Cache       4, Level 2,  256 KB, Assoc   4, LineSize  64
--------**--  Data Cache          4, Level 1,   32 KB, Assoc   8, LineSize  64
--------**--  Instruction Cache   4, Level 1,   32 KB, Assoc   8, LineSize  64
--------**--  Unified Cache       5, Level 2,  256 KB, Assoc   4, LineSize  64
----------**  Data Cache          5, Level 1,   32 KB, Assoc   8, LineSize  64
----------**  Instruction Cache   5, Level 1,   32 KB, Assoc   8, LineSize  64
----------**  Unified Cache       6, Level 2,  256 KB, Assoc   4, LineSize  64
```

再使用 `getconf -a | grep CACHE` 指令得到：

```
LEVEL1_ICACHE_SIZE                 32768
LEVEL1_ICACHE_ASSOC                8
LEVEL1_ICACHE_LINESIZE             64
LEVEL1_DCACHE_SIZE                 32768
LEVEL1_DCACHE_ASSOC                8
LEVEL1_DCACHE_LINESIZE             64
LEVEL2_CACHE_SIZE                  262144
LEVEL2_CACHE_ASSOC                 4
LEVEL2_CACHE_LINESIZE              64
LEVEL3_CACHE_SIZE                  12582912
LEVEL3_CACHE_ASSOC                 16
LEVEL3_CACHE_LINESIZE              64
LEVEL4_CACHE_SIZE                  0
LEVEL4_CACHE_ASSOC                 0
LEVEL4_CACHE_LINESIZE              0
```

前面的测试实验中，均绑定了核 2，因此 L1D Cache Size 为 $32\mathrm{KB}$，L2 Cache Size 为 $256 \mathrm{KB}$，L1D Cache Line Size 为 $64\mathrm{B}$，L1D Cache 相联度为 $8$。

后续实验将在同一台机器的 `WSL2` 环境下进行。

### 1. 测量 Cache Size

具体测试函数见 `test.cpp` 中的 `test1()` 函数。

实验中访存序列为数组循环访问，访问步长为 $16$，访存序列长度为 $1024 \times 1024 \times 512$，测试结果为：

![image-20220417232336466](https://img.wzf2000.top/image/2022/04/17/image-20220417232336466.png)

变化曲线如下：

![output](https://img.wzf2000.top/image/2022/04/17/output.png)

可以看到，在 $32\mathrm{KB}$ 和 $256\mathrm{KB}$ 附近，访存时间都有明显的跳变，这也就证明了 L1D Cache Size 为 $32\mathrm{KB}$，L2 Cache Size 为 $256 \mathrm{KB}$。

### 2. 测量 Cache Line Size

具体测试函数见 `test.cpp` 中的 `test2()` 函数。

实验中使用数组大小为 $64 \mathrm{KB}$ 大于 L1D Cache Size，访存序列长度为 $1024 \times 1024 \times 128$，测试结果与对应的访存步长见下图：

![image-20220416163409202](https://img.wzf2000.top/image/2022/04/16/image-20220416163409202.png)

变化曲线如图所示：

![output](https://img.wzf2000.top/image/2022/04/17/outputf50b21aa82b6e267.png)

可以看到，在 `block size` 为 $64\mathrm{B}$ 的时候，访存时间有明显的跳变，这也就证明了 L1D Cache Line Size 为 $64\mathrm{B}$。

### 3. 测量 Cache 相联度

具体测试函数见 `test.cpp` 中的 `test3()` 函数。

实验中使用数组大小为 $64 \mathrm{KB}$ 为 L1D Cache Size 的两倍大小，访存序列长度固定为 $1024 \times 1024 \times 64$。

访问算法与作业中给出的相同，每次访问奇数块最中间的位置 `block_size / 2`。

测试结果见下图：

![image-20220416164020780](https://img.wzf2000.top/image/2022/04/16/image-20220416164020780.png)

可以看到，在组数为 $2^n = 32$ 时，访存时间明显高于其他结果，因此可以推测相联度即为 $2^{n - 2} = 8$。

在分块数恰好为相联度的 $4$ 倍时，实际访问的块数量为相联度的 $2$ 倍，此时每当访问一个新的块对应的 Cache 中的组时，其中存放的数据必定不是当前所需的，因此几乎每次访问都会造成 Cache 缺失，导致速度最慢。

而组数较少时，会出现部分冗余的缓存；组数较多时，会导致相邻的访问可能在同一组 Cache 中，无法保证次次冲突。因此访存时间都会相对减少。

### 4. 矩阵乘优化

首先我将循环中 `j` 与 `k` 的顺序互换，这使得在最后一层循环时，三个数组访问顺序都是线性变化的（或不变）。

其次，我将 `k` 的变化每次限制在了 $16$，并将 `k` 成组的循环提到了最外面，在内部 `k` 只会变化不超过 $16$，这样在具体的某个 `i` 时，三个数组需要用到的所有空间不会超过 L1D Cache Size 即 $32 \mathrm{KB}$。

最终测试结果如下：

![image-20220416164650205](https://img.wzf2000.top/image/2022/04/16/image-20220416164650205.png)

### 5. 意见与建议

`lscpu` 指令给出的参数为多核参数，具体各个核的参数似乎不是很好得到。

本地测试过程中，即便绑定了核心以及设置了 `register`，访存速度的波动也很大，对分析会造成一定困难。

