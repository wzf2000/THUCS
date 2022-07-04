## 现代密码学第一次作业

<h4 style="text-align:right">计 93 王哲凡 2019011200</h4>

### 一、Enigma

#### 结果展示

本题我使用了 (2) 图灵法来破译，具体源码参见 `Enigma/main.py` 和 `Enigma/decoder/*.py`。

直接运行 `main.py` 即可获得结果：

```
Rotor set found!
Rotors' id: (0, 2, 1)
Rotors' initial position: ['A', 'A', 'A']
```

#### 具体过程

##### 1. 找环

首先通过宽度优先搜索，找到不含重边的环。

这一步我们在队列中保存各个 `Path` 的信息，并且规定，第一条边对应在明文密文中的位置必须最小，且必须从明文到密文。

由此我们可以找到 $5$ 个互不包含相关的最小环：

```
fetched 5 chains:
length = 3, chain = [('A', 'O', 7), ('O', 'E', 9), ('E', 'A', 16)]
length = 2, chain = [('O', 'X', 5), ('X', 'O', 13)]
length = 4, chain = [('L', 'P', 4), ('P', 'J', 6), ('J', 'F', 12), ('F', 'L', 14)]
length = 5, chain = [('B', 'P', 0), ('P', 'J', 6), ('J', 'F', 12), ('F', 'L', 14), ('L', 'B', 11)]
```

此处每条边包括了出发和到达的字母，以及在明文密文中的位置。

##### 2. 缩小转子设置可能

对于所有的可能设置（$3! \times 26^2 \times 25$），我们可以使用已知的 $5$ 个环过滤来确认是否可能。

从 `A` 到 `Z` 枚举环开头对应明文字母的插线板设置，从这点开始，根据环记录的明文密文偏移，可以模拟转子对应的状态，进而得到每条边的实际输入输出，经过完整的环后，如果转子设置合理，应当能够回到最初枚举的字母上。

如果 $26$ 个字母都无法满足要求，则说明这种转子设置不可能。

这样通过 $5$ 个最小环的限制，我们可以将转子设置缩小到较小的数量上（大约为 $6000$ 种）。

##### 3. 通过插线板情况排除和取优

在上面通过环测试的转子设置中，考虑其中可以确定的插线板配置（即只有一个字母可以成功回到），如果出现了插线板配置有问题（同一个字母映射了两次不同的）或者数量过多（超过六个），则也可以排除，最终六种转子顺序排列下分别找到了：

```
found 161 settings
rotors' order (0, 1, 2), max matched number = 6, setting = ['U', 'B', 'B']
found 166 settings
rotors' order (0, 2, 1), max matched number = 8, setting = ['A', 'A', 'A']
found 174 settings
rotors' order (1, 0, 2), max matched number = 6, setting = ['X', 'Y', 'F']
found 172 settings
rotors' order (1, 2, 0), max matched number = 7, setting = ['S', 'Z', 'X']
found 178 settings
rotors' order (2, 0, 1), max matched number = 4, setting = ['O', 'T', 'A']
found 188 settings
rotors' order (2, 1, 0), max matched number = 7, setting = ['C', 'M', 'Q']
```

而剩余的转子设置中，我们考虑其中与原文环匹配数量最大的作为结果，其中每个环如果有多种字母可能，则选取匹配数量最大的那种，并且将不同环求和相加作为最终的匹配数量。

如上显示，对于转子顺序 `I -> III -> II`，`initial position` 为 `A-A-A` 时，匹配数量达到最大，因此可认为这就是真实的转子设置，也与实际情况相符合。

### 二、书本习题

#### 1.6

代码见 `1-6.py`，可知对合密钥有 $K = 0, 13$。

考虑加密解密函数均为：
$$
e_K(x) = d_K(x) = (x + K) \bmod 26
$$
因此：
$$
d_K(e_K(x)) = (x + 2K) \bmod 26 = x
$$
于是：
$$
2K \bmod 26 = 0 \Rightarrow K = 0, 13
$$

#### 1.7

仿射密码加密函数为：
$$
e(x) = (ax + b) \bmod m
$$
其中 $a$ 与 $m$ 互质，$b = 0, 1, \cdots, m - 1$。

因此仿射密码密钥数量即为：
$$
m \cdot \varphi(m)
$$
当 $m = 30, 100, 1225$ 时分别为 $240, 4000, 1029000$。
$$
30 = 2 \times 3 \times 5, \varphi(30) = 8 \\
100 = 2^2 \times 5^2, \varphi(100) = 40 \\
1225 = 5^2 \times 7^2, \varphi(100) = 840
$$

#### 1.21

##### (a)

代码见 `1-21-a.py`。

首先统计原文各字母出现频率：

```python
{'A': 0.01953125, 'B': 0.0, 'C': 0.14453125, 'D': 0.03125, 'E': 0.046875, 'F': 0.03515625, 'G': 0.09375, 'H': 0.01953125, 'I': 0.05859375, 'J': 0.02734375, 'K': 0.0703125, 'L': 0.02734375, 'M': 0.01953125, 'N': 0.05078125, 'O': 0.0390625, 'P': 0.0234375, 'Q': 0.00390625, 'R': 0.0, 'S': 0.078125, 'T': 0.0, 'U': 0.0546875, 'V': 0.0, 'W': 0.01953125, 'X': 0.02734375, 'Y': 0.05859375, 'Z': 0.05078125}
```

可发现最高频率为 `C`，猜测为 `e`，并根据提示 `F` 解密到 `w`。

进一步统计词频（去除了被包含且频率相等的情况）：

```python
{'MG': 3, 'GL': 3, 'CG': 7, 'NC': 5, 'US': 3, 'WY': 2, 'YS': 5, 'SF': 4, 'NS': 2, 'CY': 3, 'DP': 2, 'UM': 2, 'GY': 4, 'IC': 3, 'SI': 3, 'CK': 5, 'PK': 2, 'KU': 3, 'UG': 2, 'GK': 4, 'GO': 5, 'AC': 5, 'KS': 3, 'KZ': 2, 'XE': 3, 'CJ': 3, 'SH': 3, 'XC': 3, 'ZC': 7, 'CN': 5, 'KG': 3, 'DS': 2, 'LK': 2, 'IU': 2, 'IG': 2, 'FZ': 4, 'EO': 2, 'EU': 2, 'CI': 3, 'UC': 2, 'CS': 2, 'ZE': 3, 'YSF': 3, 'CYK': 2, 'JCK': 2, 'GOL': 2, 'NCG': 2, 'GAC': 2, 'CKS': 2, 'SAC': 2, 'CKX': 2, 'KSH': 2, 'GOI': 3, 'ZCN': 2, 'KGO': 2, 'JNC': 2, 'CJU': 2, 'UZC': 2, 'ZEJ': 2, 'ICGI': 2, 'FZCCN': 3, 'CFZCCN': 2, 'FZCCNDGYYS': 2, 'ZCCNDGYYSF': 2}
```

对应解密后部分情形为：

```python
{'MG': 3, 'GL': 3, 'eG': 7, 'Ne': 5, 'US': 3, 'WY': 2, 'YS': 5, 'Sw': 4, 'NS': 2, 'eY': 3, 'DP': 2, 'UM': 2, 'GY': 4, 'Ie': 3, 'SI': 3, 'eK': 5, 'PK': 2, 'KU': 3, 'UG': 2, 'GK': 4, 'GO': 5, 'Ae': 5, 'KS': 3, 'KZ': 2, 'XE': 3, 'eJ': 3, 'SH': 3, 'Xe': 3, 'Ze': 7, 'eN': 5, 'KG': 3, 'DS': 2, 'LK': 2, 'IU': 2, 'IG': 2, 'wZ': 4, 'EO': 2, 'EU': 2, 'eI': 3, 'Ue': 2, 'eS': 2, 'ZE': 3, 'YSw': 3, 'eYK': 2, 'JeK': 2, 'GOL': 2, 'NeG': 2, 'GAe': 2, 'eKS': 2, 'SAe': 2, 'eKX': 2, 'KSH': 2, 'GOI': 3, 'ZeN': 2, 'KGO': 2, 'JNe': 2, 'eJU': 2, 'UZe': 2, 'ZEJ': 2, 'IeGI': 2, 'wZeeN': 3, 'ewZeeN': 2, 'wZeeNDGYYS': 2, 'ZeeNDGYYSw': 2}
```

观察到 `wZeeN` 单词极有可能为 `wheel`，故猜测 `Z` 解密到 `h`，`N` 解密到 `l`，可得新词频：

```python
{'MG': 3, 'GL': 3, 'eG': 7, 'le': 5, 'US': 3, 'WY': 2, 'YS': 5, 'Sw': 4, 'lS': 2, 'eY': 3, 'DP': 2, 'UM': 2, 'GY': 4, 'Ie': 3, 'SI': 3, 'eK': 5, 'PK': 2, 'KU': 3, 'UG': 2, 'GK': 4, 'GO': 5, 'Ae': 5, 'KS': 3, 'Kh': 2, 'XE': 3, 'eJ': 3, 'SH': 3, 'Xe': 3, 'he': 7, 'el': 5, 'KG': 3, 'DS': 2, 'LK': 2, 'IU': 2, 'IG': 2, 'wh': 4, 'EO': 2, 'EU': 2, 'eI': 3, 'Ue': 2, 'eS': 2, 'hE': 3, 'YSw': 3, 'eYK': 2, 'JeK': 2, 'GOL': 2, 'leG': 2, 'GAe': 2, 'eKS': 2, 'SAe': 2, 'eKX': 2, 'KSH': 2, 'GOI': 3, 'hel': 2, 'KGO': 2, 'Jle': 2, 'eJU': 2, 'Uhe': 2, 'hEJ': 2, 'IeGI': 2, 'wheel': 3, 'ewheel': 2, 'wheelDGYYS': 2, 'heelDGYYSw': 2}
```

以及部分解密结果：

```
EMGLOSUDeGDleUSWYSwHlSweYKDPUMLWGYIeOXYSIPJeKQPKUGKMGOLIeGIleGAeKSlISAeYKhSeKXEeJeKSHYSXeGOIDPKhelKSHIeGIWYGKKGKGOLDSILKGOIUSIGLEDSPWhUGwheelDGYYSwUShelXEOJleGYEOWEUPXEhGAeGlwGLKlSAeIGOIYeKXeJUeIUhewheelDGYYSwEUEKUheSOewheeleIAehEJleSHwhEJhEGMXeYHeJUMGKUeY
```

观察到 `Uhe` 出现了两次，猜测 `Uhe` 对应 `the`，其中 `U` 解密到 `t`。

进一步解密结果：

```
EMGLOStDeGDletSWYSwHlSweYKDPtMLWGYIeOXYSIPJeKQPKtGKMGOLIeGIleGAeKSlISAeYKhSeKXEeJeKSHYSXeGOIDPKhelKSHIeGIWYGKKGKGOLDSILKGOItSIGLEDSPWhtGwheelDGYYSwtShelXEOJleGYEOWEtPXEhGAeGlwGLKlSAeIGOIYeKXeJteIthewheelDGYYSwEtEKtheSOewheeleIAehEJleSHwhEJhEGMXeYHeJtMGKteY
```

观察到 `theSOewheel`，猜测对应 `the one wheel` 语义，于是 `S` 解密到 `o`，`O` 解密到 `n`。

由于 `eG` 出现了 $7$ 次，而 `Ge` 未出现，且 `G` 频率为 `0.09375`，猜测 `G` 解密到 `a`：

```python
{'Ma': 3, 'aL': 3, 'ea': 7, 'le': 5, 'to': 3, 'WY': 2, 'Yo': 5, 'ow': 4, 'lo': 2, 'eY': 3, 'DP': 2, 'tM': 2, 'aY': 4, 'Ie': 3, 'oI': 3, 'eK': 5, 'PK': 2, 'Kt': 3, 'ta': 2, 'aK': 4, 'an': 5, 'Ae': 5, 'Ko': 3, 'Kh': 2, 'XE': 3, 'eJ': 3, 'oH': 3, 'Xe': 3, 'he': 7, 'el': 5, 'Ka': 3, 'Do': 2, 'LK': 2, 'It': 2, 'Ia': 2, 'wh': 4, 'En': 2, 'Et': 2, 'eI': 3, 'te': 2, 'eo': 2, 'hE': 3, 'Yow': 3, 'eYK': 2, 'JeK': 2, 'anL': 2, 'lea': 2, 'aAe': 2, 'eKo': 2, 'oAe': 2, 'eKX': 2, 'KoH': 2, 'anI': 3, 'hel': 2, 'Kan': 2, 'Jle': 2, 'eJt': 2, 'the': 2, 'hEJ': 2, 'IeaI': 2, 'wheel': 3, 'ewheel': 2, 'wheelDaYYo': 2, 'heelDaYYow': 2}
```

```
EMaLnotDeaDletoWYowHloweYKDPtMLWaYIenXYoIPJeKQPKtaKManLIeaIleaAeKolIoAeYKhoeKXEeJeKoHYoXeanIDPKhelKoHIeaIWYaKKaKanLDoILKanItoIaLEDoPWhtawheelDaYYowtohelXEnJleaYEnWEtPXEhaAealwaLKloAeIanIYeKXeJteIthewheelDaYYowEtEKtheonewheeleIAehEJleoHwhEJhEaMXeYHeJtMaKteY
```

而 `E` 出现在了句首，且 `EtEKtheonewheel` 像是一个句子的开头，猜测 `E` 解密到 `i`，`K` 解密到 `s`：

```
iMaLnotDeaDletoWYowHloweYsDPtMLWaYIenXYoIPJesQPstasManLIeaIleaAesolIoAeYshoesXieJesoHYoXeanIDPshelsoHIeaIWYassasanLDoILsanItoIaLiDoPWhtawheelDaYYowtohelXinJleaYinWitPXihaAealwaLsloAeIanIYesXeJteIthewheelDaYYowitistheonewheeleIAehiJleoHwhiJhiaMXeYHeJtMasteY
```

根据开头的 `iMaLnot` 推测为 `I may not`，即 `M` 解密到 `m`，`L` 解密到 `y`：

```
imaynotDeaDletoWYowHloweYsDPtmyWaYIenXYoIPJesQPstasmanyIeaIleaAesolIoAeYshoesXieJesoHYoXeanIDPshelsoHIeaIWYassasanyDoIysanItoIayiDoPWhtawheelDaYYowtohelXinJleaYinWitPXihaAealwaysloAeIanIYesXeJteIthewheelDaYYowitistheonewheeleIAehiJleoHwhiJhiamXeYHeJtmasteY
```

看到结尾 `masteY` 结合 `Y` 的词频较高，猜测 `Y` 解密到 `r`：

```
imaynotDeaDletoWrowHlowersDPtmyWarIenXroIPJesQPstasmanyIeaIleaAesolIoAershoesXieJesoHroXeanIDPshelsoHIeaIWrassasanyDoIysanItoIayiDoPWhtawheelDarrowtohelXinJlearinWitPXihaAealwaysloAeIanIresXeJteIthewheelDarrowitistheonewheeleIAehiJleoHwhiJhiamXerHeJtmaster
```

结尾部分构成了 `whiJhiamXerHeJtmaster`，猜测对应原文 `which i am perfect master`，即 `J` 解密到 `c`，`X` 解密到 `p`，`H` 解密到 `f`：

```
imaynotDeaDletoWrowflowersDPtmyWarIenproIPcesQPstasmanyIeaIleaAesolIoAershoespiecesofropeanIDPshelsofIeaIWrassasanyDoIysanItoIayiDoPWhtawheelDarrowtohelpinclearinWitPpihaAealwaysloAeIanIrespecteIthewheelDarrowitistheonewheeleIAehicleofwhichiamperfectmaster
```

看到 `Aehicle` 与 `wheel` 相关，可知 `A` 解密到 `v`，而 `Wrowflowers` 可猜测 `W` 解密到 `g`：

```
imaynotDeaDletogrowflowersDPtmygarIenproIPcesQPstasmanyIeaIleavesolIovershoespiecesofropeanIDPshelsofIeaIgrassasanyDoIysanItoIayiDoPghtawheelDarrowtohelpinclearingitPpihavealwaysloveIanIrespecteIthewheelDarrowitistheonewheeleIvehicleofwhichiamperfectmaster
```

尝试解读第一句：

```
I may not be able to grow flowers.
```

可得 `D` 解读为 `b`：

```
imaynotbeabletogrowflowersbPtmygarIenproIPcesQPstasmanyIeaIleavesolIovershoespiecesofropeanIbPshelsofIeaIgrassasanyboIysanItoIayiboPghtawheelbarrowtohelpinclearingitPpihavealwaysloveIanIrespecteIthewheelbarrowitistheonewheeleIvehicleofwhichiamperfectmaster
```

剩余片段可以直接解读，得到：

```
I may not be able to grow flowers, but my garden produces just as many dead leaves, old over shoes piece of rope, and bushels of dead grass as anybody's. And today I bought a wheel barrow to help in clearing it up. I have always loved and respected the wheel barrow. It is the one wheeled vehicle of which I am perfect master.
```

##### (b)

代码见 `1-21-b.py`。

从小到大枚举频率间隔（不超过 $10$），计算重合指数：

```
length = 1
0.040871838349583155
length = 2
0.038461538461538464 0.04712004562303963
length = 3
0.055941845764854614 0.048101673101673105 0.048262548262548256
length = 4
0.03725490196078431 0.042742398164084906 0.037578886976477335 0.04905335628227194
length = 5
0.04258121158911326 0.04302019315188762 0.032564450474898234 0.035278154681139755 0.042966983265490734
length = 6
0.06265664160401002 0.08376623376623375 0.04935064935064935 0.06493506493506494 0.04285714285714286 0.07337662337662337
length = 7
0.030612244897959186 0.044326241134751775 0.04343971631205674 0.040780141843971635 0.044326241134751775 0.044326241134751775 0.040780141843971635
length = 8
0.03322259136212625 0.04065040650406504 0.03368176538908246 0.04065040650406504 0.03948896631823461 0.04529616724738676 0.04065040650406504 0.0545876887340302
length = 9
0.051209103840682786 0.04267425320056899 0.06401137980085347 0.07539118065433854 0.04054054054054054 0.03453453453453453 0.04354354354354354 0.04804804804804805 0.042042042042042045
length = 10
0.040998217468805706 0.04278074866310161 0.035650623885918005 0.0392156862745098 0.03208556149732621 0.044563279857397504 0.0338680926916221 0.028409090909090908 0.030303030303030304 0.04924242424242424
```

可发现在 $m = 6$ 时重合指数最接近 $0.065$，故推测为密钥长度 $6$。

对各子部分分别统计字母频率：

```
offset = 0: {'A': 0.05263157894736842, 'B': 0.0, 'C': 0.07017543859649122, 'D': 0.0, 'E': 0.017543859649122806, 'F': 0.03508771929824561, 'G': 0.10526315789473684, 'H': 0.03508771929824561, 'I': 0.03508771929824561, 'J': 0.10526315789473684, 'K': 0.05263157894736842, 'L': 0.0, 'M': 0.0, 'N': 0.07017543859649122, 'O': 0.0, 'P': 0.07017543859649122, 'Q': 0.12280701754385964, 'R': 0.0, 'S': 0.0, 'T': 0.07017543859649122, 'U': 0.0, 'V': 0.08771929824561403, 'W': 0.07017543859649122, 'X': 0.0, 'Y': 0.0, 'Z': 0.0}
offset = 1: {'A': 0.0, 'B': 0.0, 'C': 0.08928571428571429, 'D': 0.03571428571428571, 'E': 0.05357142857142857, 'F': 0.14285714285714285, 'G': 0.0, 'H': 0.0, 'I': 0.05357142857142857, 'J': 0.017857142857142856, 'K': 0.16071428571428573, 'L': 0.0, 'M': 0.0, 'N': 0.017857142857142856, 'O': 0.0, 'P': 0.0, 'Q': 0.0, 'R': 0.14285714285714285, 'S': 0.017857142857142856, 'T': 0.05357142857142857, 'U': 0.03571428571428571, 'V': 0.10714285714285714, 'W': 0.017857142857142856, 'X': 0.0, 'Y': 0.03571428571428571, 'Z': 0.017857142857142856}
offset = 2: {'A': 0.03571428571428571, 'B': 0.05357142857142857, 'C': 0.07142857142857142, 'D': 0.08928571428571429, 'E': 0.03571428571428571, 'F': 0.08928571428571429, 'G': 0.03571428571428571, 'H': 0.0, 'I': 0.0, 'J': 0.05357142857142857, 'K': 0.017857142857142856, 'L': 0.10714285714285714, 'M': 0.05357142857142857, 'N': 0.017857142857142856, 'O': 0.0, 'P': 0.03571428571428571, 'Q': 0.05357142857142857, 'R': 0.08928571428571429, 'S': 0.05357142857142857, 'T': 0.0, 'U': 0.0, 'V': 0.0, 'W': 0.017857142857142856, 'X': 0.0, 'Y': 0.07142857142857142, 'Z': 0.017857142857142856}
offset = 3: {'A': 0.05357142857142857, 'B': 0.017857142857142856, 'C': 0.05357142857142857, 'D': 0.14285714285714285, 'E': 0.017857142857142856, 'F': 0.0, 'G': 0.05357142857142857, 'H': 0.0, 'I': 0.03571428571428571, 'J': 0.017857142857142856, 'K': 0.0, 'L': 0.03571428571428571, 'M': 0.0, 'N': 0.03571428571428571, 'O': 0.0, 'P': 0.08928571428571429, 'Q': 0.03571428571428571, 'R': 0.017857142857142856, 'S': 0.05357142857142857, 'T': 0.16071428571428573, 'U': 0.017857142857142856, 'V': 0.03571428571428571, 'W': 0.07142857142857142, 'X': 0.05357142857142857, 'Y': 0.0, 'Z': 0.0}
offset = 4: {'A': 0.07142857142857142, 'B': 0.07142857142857142, 'C': 0.0, 'D': 0.0, 'E': 0.03571428571428571, 'F': 0.0, 'G': 0.03571428571428571, 'H': 0.08928571428571429, 'I': 0.07142857142857142, 'J': 0.0, 'K': 0.08928571428571429, 'L': 0.05357142857142857, 'M': 0.05357142857142857, 'N': 0.03571428571428571, 'O': 0.017857142857142856, 'P': 0.07142857142857142, 'Q': 0.0, 'R': 0.017857142857142856, 'S': 0.0, 'T': 0.05357142857142857, 'U': 0.017857142857142856, 'V': 0.03571428571428571, 'W': 0.03571428571428571, 'X': 0.07142857142857142, 'Y': 0.03571428571428571, 'Z': 0.03571428571428571}
offset = 5: {'A': 0.03571428571428571, 'B': 0.10714285714285714, 'C': 0.14285714285714285, 'D': 0.017857142857142856, 'E': 0.017857142857142856, 'F': 0.017857142857142856, 'G': 0.05357142857142857, 'H': 0.125, 'I': 0.05357142857142857, 'J': 0.0, 'K': 0.03571428571428571, 'L': 0.0, 'M': 0.0, 'N': 0.0, 'O': 0.07142857142857142, 'P': 0.0, 'Q': 0.017857142857142856, 'R': 0.017857142857142856, 'S': 0.14285714285714285, 'T': 0.0, 'U': 0.0, 'V': 0.05357142857142857, 'W': 0.03571428571428571, 'X': 0.0, 'Y': 0.0, 'Z': 0.05357142857142857}
```

考虑各个子串中，⾼字频（排名前十）的字⺟替换 `E` 对应的移位密码。

为了减少人工工作量，设置重合指数大于 $0.06$ 的才进行人工判断（下为被否定的）：

```
coincidence = 0.061219443266921005
text = ILEAENEDHOJTOCALPULATEGHEAMOHNTOFPNPERNERDEDFOEAROOMJHENIWNSATSCUOOLYOHMULTICLYTHEFQUARESOOTAGROFTHEJALLSBLTHECUOICCONGENTSOSTHEFLBORANDPEILINTCOMBIAEDANDQOUBLEVTYOUTUENALLBWHALFGHETOTNLFOROCENINGFSUCHAFWINDOJSANDDBORSTHRNYOUAYLOWTHROTHERUALFFOEMATCHVNGTHECATTERATHENYBUDOUBYETHEWUOLETHVNGAGAVNTOGIIEAMARTINOFEERORANQTHENYBUORDEETHEPACER

coincidence = 0.0601066836230041
text = ILEAHNEDHOMTOCALSULATEJHEAMOKNTOFPQPERNEUDEDFOHAROOMMHENIWQSATSCXOOLYOKMULTIFLYTHEIQUAREVOOTAGUOFTHEMALLSBOTHECURICCONJENTSOVTHEFLEORANDSEILINWCOMBIDEDANDTOUBLEYTYOUTXENALLEWHALFJHETOTQLFOROFENINGISUCHAIWINDOMSANDDEORSTHUNYOUABLOWTHUOTHERXALFFOHMATCHYNGTHEFATTERDTHENYEUDOUBBETHEWXOLETHYNGAGAYNTOGILEAMARWINOFEHRORANTTHENYEUORDEHTHEPAFER

coincidence = 0.06049526635580048
text = ILPARNEDSOWTOCLLCULAEETHEAXOUNTOQPAPERYEEDEDQORAROZMWHENTWASATDCHOOLJOUMULEIPLYTSESQUACEFOOTLGEOFTSEWALLDBYTHENUBICCZNTENTDOFTHEQLOORAYDCEILTNGCOMMINEDAYDDOUBWEITYOFTHENAWLOWHAWFTHETZTALFOCOPENIYGSSUCSASWINOOWSANODOORSEHENYOFALLOWEHEOTHPRHALFQORMATNHINGTSEPATTPRNTHEYYOUDOFBLETHPWHOLEEHINGARAINTORIVEAMLRGINOQERRORLNDTHEYYOUOROERTHEAAPER

coincidence = 0.06503461918892187
text = ILEARNEDHOWTOCALCULATETHEAMOUNTOFPAPERNEEDEDFORAROOMWHENIWASATSCHOOLYOUMULTIPLYTHESQUAREFOOTAGEOFTHEWALLSBYTHECUBICCONTENTSOFTHEFLOORANDCEILINGCOMBINEDANDDOUBLEITYOUTHENALLOWHALFTHETOTALFOROPENINGSSUCHASWINDOWSANDDOORSTHENYOUALLOWTHEOTHERHALFFORMATCHINGTHEPATTERNTHENYOUDOUBLETHEWHOLETHINGAGAINTOGIVEAMARGINOFERRORANDTHENYOUORDERTHEPAPER
yes
```

最终得到明文：

```
ILEARNEDHOWTOCALCULATETHEAMOUNTOFPAPERNEEDEDFORAROOMWHENIWASATSCHOOLYOUMULTIPLYTHESQUAREFOOTAGEOFTHEWALLSBYTHECUBICCONTENTSOFTHEFLOORANDCEILINGCOMBINEDANDDOUBLEITYOUTHENALLOWHALFTHETOTALFOROPENINGSSUCHASWINDOWSANDDOORSTHENYOUALLOWTHEOTHERHALFFORMATCHINGTHEPATTERNTHENYOUDOUBLETHEWHOLETHINGAGAINTOGIVEAMARGINOFERRORANDTHENYOUORDERTHEPAPER
```

即：

```
I learned how to calculate the amount of paper needed for a room when i was at school. You multiply the square footage of the walls by the cubic contents of the floor and ceiling combined and double it. You then allow half the total for openings such as windows and doors. Then you allow the other half for matching the pattern. Then you double the whole thing again to give a margin of error and then you order the paper.
```

##### (c)

计算字母频率：

```python
{'A': 0.06565656565656566, 'B': 0.10606060606060606, 'C': 0.16161616161616163, 'D': 0.045454545454545456, 'E': 0.06565656565656566, 'F': 0.050505050505050504, 'G': 0.0, 'H': 0.005050505050505051, 'I': 0.08080808080808081, 'J': 0.030303030303030304, 'K': 0.10101010101010101, 'L': 0.0, 'M': 0.0, 'N': 0.005050505050505051, 'O': 0.010101010101010102, 'P': 0.10101010101010101, 'Q': 0.020202020202020204, 'R': 0.06060606060606061, 'S': 0.005050505050505051, 'T': 0.0, 'U': 0.030303030303030304, 'V': 0.020202020202020204, 'W': 0.0, 'X': 0.010101010101010102, 'Y': 0.005050505050505051, 'Z': 0.020202020202020204}
```

可知 `C` 和 `B` 词频最高，猜测 `C` 解密为 `e`，`B` 解密为 `t`。

假设仿射变换为：
$$
e(x) = (ax + b) \bmod 26
$$
则：
$$
\begin{cases}
(4a + b) \bmod 26 = 2 \\
(19a + b) \bmod 26 = 1
\end{cases}
$$
可得：
$$
15a \bmod 26 = 25 \Rightarrow a = (15^{-1}_{26} \times 25) \bmod 26 = (7 \times 25) \bmod 26 = 19 \\
\Rightarrow b = 4
$$
故解密函数为：
$$
d(x) = a^{-1}_{26}(x - b) \bmod 26 = 11(x - 4) \bmod 26
$$
解密得到明文：

```
OCANADATERREDENOSAIEUXTONFRONTESTCEINTDEFLEURONSGLORIEUXCARTONBRASSAITPORTERLEPEEILSAITPORTERLACROIXTONHISTOIREESTUNEEPOPEEDESPLUSBRILLANTSEXPLOITSETTAVALEURDEFOITREMPEEPROTEGERANOSFOYERSETNOSDROITS
```

即为法语版的加拿大国歌：

```
O Canada
Terre de nos aieux
Ton front est ceint de fleurons glorieux!
Car ton bras sait porter l'epee
il sait porter la croix!
Ton histoire est une epopee
Des plus brillants exploits
Et ta valeur, de foi trempee
Protegera nos foyers et nos droits
```

##### (d)

计算密文的各重复指数：

```
length = 1
0.041612614949984146
length = 2
0.04433327583232707 0.04638186573670444
length = 3
0.044645161290322574 0.04812483608707054 0.04838709677419355
length = 4
0.043468313886982385 0.05750350631136045 0.04651706404862085 0.047452080411407194
length = 5
0.04504504504504505 0.04252252252252252 0.04144144144144144 0.04405775638652351 0.03702332469455757
length = 6
0.05120327700972862 0.06345848757271286 0.05499735589635114 0.06980433632998413 0.05552617662612375 0.06980433632998413
length = 7
0.04053109713487072 0.04472396925227114 0.04426705370101596 0.03773584905660377 0.04644412191582003 0.031930333817126275 0.04644412191582003
length = 8
0.058279370952821465 0.05920444033302497 0.050878815911193344 0.0481036077705828 0.037927844588344126 0.06473429951690822 0.0463768115942029 0.050241545893719805
length = 9
0.04878048780487805 0.04529616724738676 0.04181184668989547 0.036004645760743324 0.04024390243902439 0.04024390243902439 0.04268292682926829 0.045121951219512194 0.06219512195121951
length = 10
0.06543385490753911 0.04125177809388336 0.02418207681365576 0.04354354354354354 0.04054054054054054 0.04054054054054054 0.05405405405405406 0.04954954954954955 0.052552552552552555 0.05105105105105105
```

发现 `length = 6` 时，值最为接近 $0.065$，故推测为密钥长度 $6$ 的 Vigenere 密码。

类比 (b) 解码，最终得到明文：

```
coincidence = 0.06396840496987517
IGREWUPAMONGSLOWTALKERSMENINPARTICULARWHODROPPEDWORDSAFEWATATIMELIKEBEANSINAHILLANDWHENIGOTTOMINNEAPOLISWHEREPEOPLETOOKALAKEWOBEGONCOMMATOMEANTHEENDOFASTORYICOULDNTSPEAKAWHOLESENTENCEINCOMPANYANDWASCONSIDEREDNOTTOOBRIAHTSOIENROLLEDINASPEECHCOUQSETAUGHTBYORVILLESANDTHEFOUNDEROFREFLEXIVERELAXOLOGYASELFHYPNOTICTECHNIQUETHATENABLEDAPERSONTOSPEAKUPTOTHREEHUNDREDWORDSPERMINUTE
```

也即：

```
I grew up among slow talkers' men in particular who dropped words a few at a time like beans in a hill and when I got to Minneapolis where people took a lake wobegon comma to mean the end of a story. I couldn't speak a whole sentence in company and was considered not to Obriaht. So I enrolled in a speech couqse taught by Orvilles and the founder of reflexive relaxology a self-hypnotic technique that enabled a person to speak up to three hundred words per minute.
```

其中应有 `couqse` 与 `Obriaht` 可能拼写错误。

#### 1.24

代码见 `1-24.py`。

根据明文和密文，我们可以列出 $18$ 个同余方程组，而未知数共有 $3^2 + 3 = 12$ 个，因此我们可以对这个方程组进行高斯消元。

注意到模数为 $26$，于是我们在挑选主元列时，优先考虑值为奇数的，如果均为偶数，则除法时先同时除以 $2$ 再取逆元。

这样我们可以在模 $26$ 意义下消元得到所有的密钥，即为：
$$
L = \begin{bmatrix}
3 & 6 & 4 \\
5 & 15 & 18 \\
17 & 8 & 5
\end{bmatrix}
, b = (8, 13, 1)
$$

#### 1.26

##### (a)

已知 $m$ 和 $n$ 的情况下，只需要使用矩阵转置即可解密。

先将密文按照 $m \times n$ 分组，然后各组按矩阵内顺序排列得到矩阵，再将矩阵转置并恢复文本即可获得明文。

##### (b)

代码见 `1-26.py`

枚举得到 $m = 3, n = 2$：

```
marymaryquitecontraryhowdoesyourgardengrow
```

即：

```
Mary, Mary, quite contrary, how does your garden grow?
```

