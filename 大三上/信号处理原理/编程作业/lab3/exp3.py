import numpy as np
import matplotlib.pyplot as plt

# 时长为1秒
t = 1
# 采样率为60hz
fs = 60
t_split = np.arange(0, t * fs)


# 1hz与25hz叠加的正弦信号
x_1hz = t_split * 1 * np.pi * 2 / fs
x_25hz = t_split * 25 * np.pi * 2 / fs
signal_sin_1hz = np.sin(x_1hz)
signal_sin_25hz = np.sin(x_25hz)

signal_sin = signal_sin_1hz + 0.25 * signal_sin_25hz


# TODO: 补全这部分代码
# 通带边缘频率为10Hz，
# 阻带边缘频率为22Hz，
# 阻带衰减为44dB，窗内项数为17的汉宁窗函数
# 构建低通滤波器
# 函数需要返回滤波后的信号
def filter_fir(input):
    omega_c = (22 + 10) / 2 / 60 * 2 * np.pi
    w_n = np.array([0.5 * (1 + np.cos(2 * np.pi * (n - 8) / 16)) * (omega_c / np.pi  if n == 8 else np.sin(omega_c * (n - 8)) / (np.pi * (n - 8))) for n in range(17)])
    input_pad = np.array(input.tolist() + [0] * len(w_n))
    w_n_pad = np.array(w_n.tolist() + [0] * len(input))
    a = np.fft.fft(input_pad)
    b = np.fft.fft(w_n_pad)
    output = a * b
    output = np.real(np.fft.ifft(output))
    return output[:len(input)]


# TODO: 首先正向对信号滤波(此时输出信号有一定相移)
# 将输出信号反向，再次用该滤波器进行滤波
# 再将输出信号反向
# 函数需要返回零相位滤波后的信号
def filter_zero_phase(input):
    return filter_fir(filter_fir(input)[::-1])[::-1]

if __name__ == "__main__":
    delay_filtered_signal = filter_fir(signal_sin)
    zerophase_filtered_signal = filter_zero_phase(signal_sin)

    plt.plot(t_split, signal_sin, label = 'origin')
    plt.plot(t_split, delay_filtered_signal, label = 'fir')
    plt.plot(t_split, zerophase_filtered_signal, label = 'zero phase')

    plt.legend()
    plt.show()
