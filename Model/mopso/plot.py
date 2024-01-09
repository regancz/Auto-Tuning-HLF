# encoding: utf-8
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import Axes3D
import time

from scipy.interpolate import griddata


class Plot_pareto:
    def __init__(self):
        self.start_time = time.time()

    def show(self, in_, fitness_, archive_in, archive_fitness, i):
        # 共3个子图，第1、2/子图绘制输入坐标与适应值关系，第3图展示pareto边界的形成过程
        # fig = plt.figure('第' + str(i + 1) + '次迭代')
        # fig = plt.figure('第' + str(i + 1) + '次迭代', figsize=(17, 5))
        # ax1 = fig.add_subplot(131, projection='3d')
        # ax1.set_xlabel('input_x1')
        # ax1.set_ylabel('input_x2')
        # ax1.set_zlabel('fitness_y1')
        # ax1.plot_surface(self.x1, self.x2, self.y1, alpha=0.6)
        # ax1.scatter(in_[:, 0], in_[:, 1], fitness_[:, 0], s=20, c='blue', marker=".")
        # ax1.scatter(archive_in[:, 0], archive_in[:, 1], archive_fitness[:, 0], s=50, c='red', marker=".")
        # ax2 = fig.add_subplot(132, projection='3d')
        # ax2.set_xlabel('input_x1')
        # ax2.set_ylabel('input_x2')
        # ax2.set_zlabel('fitness_y2')
        # ax2.plot_surface(self.x1, self.x2, self.y2, alpha=0.6)
        # ax2.scatter(in_[:, 0], in_[:, 1], fitness_[:, 1], s=20, c='blue', marker=".")
        # ax2.scatter(archive_in[:, 0], archive_in[:, 1], archive_fitness[:, 1], s=50, c='red', marker=".")
        #
        # ax3 = fig.add_subplot(111)  # 133
        # # ax3.set_xlim((0,1))
        # # ax3.set_ylim((0,1))
        # ax3.set_xlabel('fitness_y1')
        # ax3.set_ylabel('fitness_y2')
        # ax3.scatter(fitness_[:, 0], fitness_[:, 1], s=10, c='blue', marker=".")
        # ax3.scatter(archive_fitness[:, 0], archive_fitness[:, 1], s=30, c='red', marker=".", alpha=1.0)
        # plt.show()
        # # plt.savefig('./img_txt/'+str(i+1)+'.png')
        # # print ('第'+str(i+1)+'次迭代的图片保存于 img_txt 文件夹')
        # # print ('第'+str(i+1)+'次迭代, time consuming: ',np.round(time.time() - self.start_time, 2), "s")
        # plt.ion()
        # # plt.close()

        # 创建3D图
        # fig = plt.figure()
        # ax = fig.add_subplot(projection='3d')
        # # 绘制粒子的位置
        # ax.scatter(fitness_[:, 0], fitness_[:, 1], fitness_[:, 2], s=10, c='blue', marker=".")
        # ax.scatter(archive_fitness[:, 0], archive_fitness[:, 1], archive_fitness[:, 2], s=30, c='red', marker=".",
        #            alpha=1.0)
        # ax.set_xlabel('throughput')
        # ax.set_ylabel('avg_latency')
        # ax.set_zlabel('disc_write')
        # plt.show()
        # # animation = create_animation(fitness_)
        # # animation.save('particle_animation.gif', writer='pillow')
        # # plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # 绘制点
        ax.scatter(fitness_[:, 0], fitness_[:, 1], fitness_[:, 2], s=10, c='blue', marker=".")
        ax.scatter(archive_fitness[:, 0], archive_fitness[:, 1], archive_fitness[:, 2], s=30, c='red', marker=".",
                   alpha=1.0)

        # 设置坐标轴标签
        ax.set_xlabel('throughput')
        ax.set_ylabel('avg_latency')
        ax.set_zlabel('disc_write')

        # 对fitness_进行插值并绘制曲面
        x = fitness_[:, 0]
        y = fitness_[:, 1]
        z = fitness_[:, 2]

        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method='cubic')  # 使用cubic插值方法进行插值

        # 绘制曲面
        ax.plot_surface(xi, yi, zi, cmap='viridis', alpha=0.5)  # 以较低的alpha值绘制曲面，以便看到散点

        plt.show()


class Designer:
    def __init__(self, limits, label, colormap, figsize=(8, 6), title_fontsize=14, text_fontsize=10):
        self.limits = limits
        self.label = label
        self.colormap = colormap
        self.figsize = figsize
        self.title_fontsize = title_fontsize
        self.text_fontsize = text_fontsize


class Animator:
    def __init__(self, interval=200, repeat=True, repeat_delay=1000):
        self.interval = interval
        self.repeat = repeat
        self.repeat_delay = repeat_delay


# 动画函数
def _animate(frame, pos_history, plot):
    plot._offsets3d = (pos_history[frame:(frame + 10), 0],
                       pos_history[frame * 10:(frame + 10), 1],
                       pos_history[frame * 10:(frame + 10), 2])
    return plot,


# 创建动画
def create_animation(pos_history):
    try:
        # 如果没有提供Designer和Animator类，则使用默认值
        designer = Designer(
            limits=[(np.min(pos_history[:, 0]), np.max(pos_history[:, 0])),
                    (np.min(pos_history[:, 1]), np.max(pos_history[:, 1])),
                    (np.min(pos_history[:, 2]), np.max(pos_history[:, 2]))],
            label=["x-axis", "y-axis", "z-axis"],
            colormap='viridis',
        )

        animator = Animator()

        fig = plt.figure(figsize=designer.figsize)
        ax = fig.add_subplot(111, projection='3d')

        n_iters = len(pos_history)
        ax.set_xlim(designer.limits[0])
        ax.set_ylim(designer.limits[1])
        ax.set_zlim(designer.limits[2])

        plot = ax.scatter(xs=pos_history[:, 0],
                          ys=pos_history[:, 1],
                          zs=pos_history[:, 2], c="black",
                          alpha=0.6)

        anim = animation.FuncAnimation(
            fig=fig,
            func=_animate,
            frames=range(n_iters // 10),
            fargs=(pos_history, plot),
            interval=animator.interval,
            repeat=animator.repeat,
            repeat_delay=animator.repeat_delay,
        )

        # 返回动画变量
        return anim
    except TypeError:
        raise
