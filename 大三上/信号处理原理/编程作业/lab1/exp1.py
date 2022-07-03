import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import imageio
from matplotlib.pyplot import cm

# TODO: 1. Change N_Fourier to 2, 4, 8, 16, 32, 64, 128, get visualization results with differnet number of Fourier Series
N_Fourier = 128

# TODO: optional, implement visualization for semi-circle
# signal_name = "square"
signal_name = "semicircle"

# TODO: 2. Please implement the function that calculates the Nth fourier coefficient
# Note that n starts from 0
# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am. 
semicircle_fc = [0] * (2 * N_Fourier + 1)
def fourier_coefficient(n):
    if signal_name == "square":
        if n == 0:
            return 0.5
        elif n % 2 == 1:
            if (n + 1) // 2 % 2 == 0:
                return 0
            else:
                return 2 / ((n + 1) // 2) / np.pi
        else:
            return 0
    elif signal_name == "semicircle":
        if semicircle_fc[n] != 0:
            return semicircle_fc[n]
        N = 50000
        dt = 2 * np.pi / N
        sum = 0
        for i in range(N):
            t = 2 * np.pi / N * i + np.pi / N
            if n == 0:
                ft = 1
            elif n % 2 == 0:
                ft = np.cos(n / 2 * t)
            else:
                ft = np.sin((n + 1) / 2 * t)
            sum += dt * semi_circle_wave(t) * ft
        if n == 0:
            semicircle_fc[n] = sum / (2 * np.pi)
            return sum / (2 * np.pi)
        else:
            semicircle_fc[n] = sum / np.pi
            return sum / np.pi
    else:
        raise Exception("Unknown Signal")

# TODO: 3. implement the signal function
def square_wave(t):
    sint = np.sin(t)
    if sint >= 0:
        return 1
    else:
        return 0

# TODO: optional. implement the semi circle wave function
def semi_circle_wave(t):
    return np.sqrt(np.pi ** 2 - (t - np.pi) ** 2)

def function(t):
    if signal_name == "square":
        return square_wave(t)
    elif signal_name == "semicircle":
        return semi_circle_wave(t)
    else:
        raise Exception("Unknown Signal")


def visualize():
    if not os.path.exists(signal_name):
        os.makedirs(signal_name)

    frames = 100

    # x and y are for drawing the original function
    x = np.linspace(0, 2 * math.pi, 1000)
    y = np.zeros(1000, dtype = float)
    for i in range(1000):
        y[i] = function(x[i])

    for i in range(frames):
        figure, axes = plt.subplots()
        color=iter(cm.rainbow(np.linspace(0, 1, 2 * N_Fourier + 1)))

        time = 2 * math.pi * i / 100
        point_pos_array = np.zeros((2 * N_Fourier + 2, 2), dtype = float)
        radius_array = np.zeros((2 * N_Fourier + 1), dtype = float)

        point_pos_array[0, :] = [0, 0]
        radius_array[0] = fourier_coefficient(0)
        point_pos_array[1, :] = [0, radius_array[0]]

        circle = patches.Circle(point_pos_array[0], radius_array[0], fill = False, color = next(color))
        axes.add_artist(circle)

        f_t = function(time)
        for j in range(N_Fourier):
            # calculate circle for a_{n}
            radius_array[2 * j + 1] = fourier_coefficient(2 * j + 1)
            point_pos_array[2 * j + 2] = [point_pos_array[2 * j + 1][0] + radius_array[2 * j + 1] * math.cos((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 1][1] + radius_array[2 * j + 1] * math.sin((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 1], radius_array[2 * j + 1], fill = False, color = next(color))
            axes.add_artist(circle)
            
            # calculate circle for b_{n}
            radius_array[2 * j + 2] = fourier_coefficient(2 * j + 2)
            point_pos_array[2 * j + 3] = [point_pos_array[2 * j + 2][0] + radius_array[2 * j + 2] * math.sin((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 2][1] + radius_array[2 * j + 2] * math.cos((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 2], radius_array[2 * j + 2], fill = False, color = next(color))
            axes.add_artist(circle)
            
        plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-')
        plt.plot(x, y, '-')
        plt.plot([time, point_pos_array[-1][0]], [f_t, point_pos_array[-1][1]], '-', color = 'r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(signal_name, "{}.png".format(i)))
        # plt.show()
        plt.close()
        
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(signal_name, "{}.png".format(i))))
    imageio.mimsave('{}_n={}.mp4'.format(signal_name, N_Fourier), images)


if __name__ == "__main__":
    visualize()