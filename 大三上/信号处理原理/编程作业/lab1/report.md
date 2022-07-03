### 第一次实验

<div style="float: none">
<font style="float: right">计 93 王哲凡 2019011200</font>
</div>

#### 任务一：可视化方波信号

方波信号：
$$
f(t) = 0.5 \, \mathrm{sgn}(\sin(t)) + 0.5
$$
按照定义可求其傅里叶展开系数：
$$
\begin{align*}
a_n & = \frac{2}{2\pi} \int_0^{2pi} f(t) \cos(nt) \, dt \\
& = \frac1 \pi \int_0^{\pi} \cos (nt) \, dt \\
& = 
\begin{cases}
1, & n = 0, \\
0, & \text{otherwises}.
\end{cases}
\\
b_n & = \frac{2}{2\pi} \int_0^{2pi} f(t) \sin(nt) \, dt \\
& = \frac1 \pi \int_0^{\pi} \sin (nt) \, dt \\
& =
\begin{cases}
0, & n = 2k, k \in N^*, \\
\dfrac{2}{n \pi}, & n = 2k - 1, k \in N^*.
\end{cases}
\end{align*}
$$
因此只须按照计算结果，填入不同情况下 $a_n, b_n$ 的值即可。

运行代码可以观察到，随着 $N_{fourier}$（展开项数）的增大，最终小点的纵坐标与实际方波的值越来越接近。

#### 任务二：可视化半圆波信号（选做）

半圆波信号：
$$
f(t) = \sqrt{\pi^2 - (t - \pi)^2}, t \in [0, 2\pi] \\
f(t) = f(t - 2\pi), t \notin [0, 2\pi]
$$
由于较难以求得半圆波信号的傅里叶级数解析解，于是采用数值解法。

取 $N = 50000$，将 $[0, 2\pi]$ 均分为 $N$ 段，利用各段矩形面积和逼近积分：
$$
\begin{align*}
a_n & = \frac{2}{2\pi} \int_0^{2pi} f(t) \cos(nt) \, dt \\
& \approx \frac{1}{\pi} \frac{2 \pi}{N} \sum_{i = 0}^{N - 1} f\left(\frac{2\pi}{N} *(i + 0.5)\right) \cos\left(n\frac{2\pi}{N} *(i + 0.5)\right) \\
b_n & = \frac{2}{2\pi} \int_0^{2pi} f(t) \sin(nt) \, dt \\
& \approx \frac{1}{\pi} \frac{2 \pi}{N} \sum_{i = 0}^{N - 1} f\left(\frac{2\pi}{N} *(i + 0.5)\right) \sin\left(n\frac{2\pi}{N} *(i + 0.5)\right)
\end{align*}
$$
为了加快计算，记录每个 $a_n, b_n$ 第一次计算后的结果，在第二次调用时直接返回缓存结果：

```python
if semicircle_fc[n] != 0:
	return semicircle_fc[n]
```

运行代码可以观察到，当 $N_{fourier}$ 增大，小点的纵坐标与实际方波的值越来越接近。

