# coding: utf-8
import math
import os
import random
from typing import List, Tuple

import matplotlib.gridspec
# matplotlib.use("Agg") # used for writing to a file?
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

def find_csv(path):
    file_list = []
    for root, dirs, files in os.walk(path):
        for filename in files:
            if os.path.splitext(filename)[1] == '.csv':
                file_list.append(os.path.join(root, filename))

    return file_list


class ScatterPlot:

    def __init__(self, name_a="tool a", name_b="tool b"):
        rcParams['font.family'] = 'serif'
        rcParams['font.serif'] = ['Times New Roman']
        rcParams["text.usetex"] = False

        # rcParams['pdf.fonttype'] = 42
        # rcParams['ps.fonttype'] = 42

        self.m_upper_bound = 1  # upper-bound
        self.m_use_log_scale = False  # log-scale
        self.tool_a_name = name_a
        self.tool_b_name = name_b

    def get_scatter_plot(self, data: Tuple[List, List], output_dir="", filename="test", save_to_file=False):
        """
        Scatter Plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        x, y = data
        if self.m_use_log_scale:
            x = [math.log(i, 10) for i in x]
            y = [math.log(i, 10) for i in y]

        ax.scatter(x, y, alpha=0.5, marker="x", c="blue")

        ax.set_xlabel("Result of {}".format(self.tool_a_name), fontsize=11)
        ax.set_ylabel("Result of {}".format(self.tool_b_name), fontsize=12)

        bound = self.m_upper_bound
        if self.m_use_log_scale: bound = math.log(self.m_upper_bound, 10)

        plt.plot([0, bound], [0, bound], 'k', linewidth=0.7)
        plt.title('')
        plt.show()
        if save_to_file:
            out_file = os.path.join(output_dir, filename + "-scatter.png")
            print("to file:, ", filename + "-scatter.png")

            plt.savefig(out_file, dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, pad_inches=0.02, bbox_inches='tight')

            plt.close()

    def get_scatter_plot_multi_groups(self, data: Tuple[Tuple[List, List], Tuple[List, List]], output_dir="",
                                      filename="test",
                                      save_to_file=False):
        """
        Scatter Plot
        """
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # print(data)

        colors = ("blue", "green")
        groups = ("sat", "unsat")
        for data, color, group in zip(data, colors, groups):
            print(data, color, group)
            x, y = data
            if self.m_use_log_scale:
                x = [math.log(i, 10) for i in x]
                y = [math.log(i, 10) for i in y]

            if color == "blue":
                ax.scatter(x, y, alpha=0.5, marker="x", c="blue")
            else:
                ax.scatter(x, y, alpha=0.5, marker="s", c="green")

        ax.set_xlabel("Result of {}".format(self.tool_a_name), fontsize=11)
        ax.set_ylabel("Result of {}".format(self.tool_b_name), fontsize=12)

        bound = self.m_upper_bound
        if self.m_use_log_scale: bound = math.log(self.m_upper_bound, 10)

        plt.plot([0, bound], [0, bound], 'k', linewidth=0.7)
        plt.title('')
        plt.show()
        if save_to_file:
            out_file = os.path.join(output_dir, filename + "-scatter.png")
            print("to file:, ", filename + "-scatter.png")

            plt.savefig(out_file, dpi=300, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, pad_inches=0.02, bbox_inches='tight')

            plt.close()


class CactusPlot:

    def __init__(self):
        user_fontsize = 12
        font = {'size': user_fontsize}
        matplotlib.rc('font', **font)

        self.m_use_log_scale = False  # log-scale
        self.m_upper_bound = 9  # cut-off

    def get_cactus_plot(self, data: List[List], out_dir="", filename="test", save_to_file=False):
        num_of_tools = len(data)

        gs = matplotlib.gridspec.GridSpec(1, 1)

        # fig = plt.figure()
        ax1 = plt.subplot(gs[0:1, :])

        # remove timeouts
        data_after_clean = []
        for s in data:
            data_after_clean.append(list(filter(lambda x: True if x < self.m_upper_bound else False, s)))

        res = [[] for _ in range(num_of_tools)]
        cur = [0 for _ in range(num_of_tools)]

        for i in range(num_of_tools):
            s = data_after_clean[i]
            for j in range(len(s)):
                cur[i] = cur[i] + s[j]
                res[i].append(cur[i])

        if self.m_use_log_scale:
            res_final = []
            for r in res:
                res_final.append([math.log(2 * i, 10) for i in r])
        else:
            res_final = res

        print("Data processed. Begin to dot")

        color_list = []
        marker_list = []
        label_list = []

        all_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        all_markers = ['.', ',', 'o', '<', '>', 's', 'p', '*', 'x', 'h', 'H',
                       'D', '|', '1', '2', '3', '4']

        for i in range(num_of_tools):
            color_list.append(random.choice(all_colors))
            marker_list.append(random.choice(all_markers))
            label_list.append(str(i) + "-th tool")

        for i in range(num_of_tools):
            ax1.plot(res_final[i], color=color_list[i], marker=marker_list[i], label=label_list[i], markevery=3)

        ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
        ax1.set_xlabel("#solved instances")
        if self.m_use_log_scale:
            ax1.set_ylabel("Runtime [sec] log scale")
        else:
            ax1.set_ylabel("Runtime [sec] ")

        max_list = []
        for r in res_final:
            max_list.append(r[-1])
        ax1.set_ylim([0, max(max_list)])

        ax1.legend(loc='lower right')

        plt.show()

        if save_to_file:
            out_file = os.path.join(out_dir, filename + "-cactus.pdf")
            plt.savefig(out_file, dpi=100, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format=None,
                        transparent=False, pad_inches=0.02, bbox_inches='tight')
            plt.close()


class BoxPlot:

    def __init__(self):
        self.name = ""

    def get_box_plot(self, data: List[List], save_to_file=False):
        """
        data = [[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]]
        """
        assert len(data) > 0

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
        bplot = axes.boxplot(data,
                             vert=True,
                             patch_artist=True)

        all_colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'red']
        assert len(data) <= len(all_colors)
        colors = random.sample(all_colors, len(data))

        # colors = ['pink', 'lightblue', 'lightgreen']

        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

        # axes.yaxis.grid(True)

        axes.set_xlabel('xlabel')  #
        axes.set_ylabel('ylabel')  #

        plt.setp(axes, xticks=[i + 1 for i in range(len(data))],
                 xticklabels=['x' + str(i) for i in range(len(data))]
                 )

        plt.show()

    def get_box_plot_multi_groups(self, all_data: List[List[List]], save_to_file=False):
        """
        Draw multiple figures

        all_data = []
        all_data.append([[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]])
        all_data.append([[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]])
        """
        assert len(all_data) > 0

        # 1 X N (NOTE: we can use A X B)
        fig, axes = plt.subplots(nrows=1, ncols=len(all_data), figsize=(9, 4))

        bplots = []

        for i in range(len(all_data)):
            bplots.append(axes[i].boxplot(all_data[i],
                                          vert=True,
                                          patch_artist=True))

        all_colors = ['pink', 'lightblue', 'lightgreen', 'lightyellow', 'red']
        assert len(all_data[0]) <= len(all_colors)
        colors = random.sample(all_colors, len(all_data[0]))

        # colors = ['pink', 'lightblue', 'lightgreen']
        for bplot in tuple(bplots):
            for patch, color in zip(bplot['boxes'], colors):
                patch.set_facecolor(color)

        for ax in axes:
            # ax.yaxis.grid(True)  #
            ax.set_xlabel('xlabel')
            ax.set_ylabel('ylabel')

        plt.setp(axes, xticks=[i + 1 for i in range(len(all_data[0]))],
                 xticklabels=['x' + str(i) for i in range(len(all_data[0]))]
                 )

        plt.show()


def test_plot():
    interval_fp = [0.4, 0.6, 0.33, 0.2]
    octagon_fp = [0.3, 0.7, 0.6, 0.33]

    data = (interval_fp, octagon_fp)

    sc = ScatterPlot()
    sc.get_scatter_plot(data)


def test_plot_multi_groups():
    """
    Two groups: sat and unsat
    """
    sol1_sat = [2, 3, 4, 8]
    sol2_sat = [1, 5, 6, 6]
    sol1_unsat = [2, 3, 4, 5]
    sol2_unsat = [1, 2, 6, 8]
    g1 = (sol1_sat, sol2_sat)
    g2 = (sol1_unsat, sol2_unsat)

    # data = (g1, g2, g3)
    data = (g1, g2)
    sc = ScatterPlot()
    sc.get_scatter_plot(data)


def test_plot_cactus():
    s1 = [1, 2, 3, 3.5, 2.5, 4, 6]
    s2 = [5, 2, 2, 5, 8, 2, 10]
    s3 = [1, 7, 3, 3, 2, 4, 10, 12, 15]
    s4 = [5, 6, 1, 2, 3, 8, 7, 9, 12, 10]
    # s5 = [5, 6, 1, 2, 3, 8, 3, 9, 10, 10]
    # get_scatter_plot(s2, s4, "", 9)
    cc = CactusPlot()
    cc.get_cactus_plot([s1, s2, s3, s4])


def test_plot_pinpoint():
    arrLoc = [18.6, 22.4, 24.9, 25.2, 44.3, 54.4, 68.1, 76.4, 85.6, 89.5, 196.5, 384.2]
    arrTimeW = [0.36, 0.21, 0.71, 0.77, 0.62, 1.35, 1.63, 0.91, 2.72, 1.65, 6.83, 13.21]
    arrMemW = [0.93, 0.78, 1.18, 1.25, 1.21, 1.74, 2.18, 1.52, 2.67, 1.71, 4.59, 9.36]

    arrNum = len(arrLoc)
    arrRange = range(arrNum)

    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4))

    ax2.scatter(arrLoc, arrTimeW, marker='o', color='chocolate', label="time", linewidth=1, zorder=1)
    ax2.scatter(arrLoc, arrMemW, marker='^', color='dodgerblue', label="memory", linewidth=1, zorder=1)
    ax2.set_ylabel("Time(min)/Memory(G)")
    ax2.set_xlabel("Size(KLoC)")
    ax2.legend(loc="best")
    # ax2.set_title("FalconS")

    z_ts = np.polyfit(arrLoc, arrTimeW, 1)
    p_ts = np.poly1d(z_ts)
    arrTimeS2 = p_ts(arrLoc)
    plt.plot(arrLoc, arrTimeS2, 'k-')
    print("Time:")
    time_correlation = np.corrcoef(arrTimeW, arrTimeS2)[0, 1]
    print(p_ts)
    print(time_correlation ** 2)

    z_ms = np.polyfit(arrLoc, arrMemW, 1)
    p_ms = np.poly1d(z_ms)
    plt.plot(arrLoc, p_ms(arrLoc), 'k--')
    print("Memory:")
    memory_correlation = np.corrcoef(arrMemW, p_ms(arrLoc))[0, 1]
    print(p_ms)
    print(memory_correlation ** 2)

    plt.text(150, 9.5, '$y = 0.03594 \; x - 0.6844$')
    plt.text(150, 8.5, '$R^2 = 0.9796$')

    plt.text(250, 4, '$y = 0.02285 \; x + 0.3509$')
    plt.text(250, 3, '$R^2 = 0.9786$')

    plt.axis("auto")
    plt.show()
    # plt.savefig("../../output/TimeMemory_FalconS.eps", dpi=600, format='eps', bbox_inches='tight')
    # plt.savefig("../../output/TimeMemory_FalconS.jpg", format='jpg', bbox_inches='tight')
    # print("save output/TimeMemory_FalconS.jpg")


def test_plot_bar():
    projects = ('1',
                '2',
                '3',
                '4',
                '5',
                '6',
                '7',
                '8',
                '9',
                '10',
                '11',
                '12',
                '13',
                '14',
                '15',
                '16'
                '17',
                '18')

    falconS_ratio = [
        0.570552147,
        0.677892312,
        0.501915709,
        0.439710987,
        0.870689655,
        0.541666667,
        0.324786325,
        0.759187631,
        0.641176471,
        0.60371517,
        0.707762557,
        0.489878543,
        0.489082969,
        0.357746479,
        0.764957265,
        0.321243523,
        0.492822967,
        0.448680352]

    falconW_ratio = [
        0.665644172,
        0.741278901,
        0.620689655,
        0.521289768,
        1.004310345,
        0.729166667,
        0.431798976,
        0.871976194,
        0.802941176,
        0.775665635,
        0.98109589,
        0.599190283,
        0.672489083,
        0.456338028,
        0.892991453,
        0.388963731,
        0.627464115,
        0.539442815
    ]

    avg = 0
    x = list(range(18))
    ratio = []
    for i in x:
        print(i)
        ratio.append(1 - falconS_ratio[i] / falconW_ratio[i])
        avg += 1 - falconS_ratio[i] / falconW_ratio[i]
    total_width, n = 0.7, 1
    width = total_width / (1.5 * n)
    plt.figure(figsize=(9, 4))
    plt.xlim((0, 19))
    plt.ylim((0, 0.4))

    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 18,
             }

    plt.xlabel('Project ID', font2)
    plt.ylabel('Decrease Ratio', font2)

    for i in range(len(x)):
        x[i] = x[i] + width * 2.5 - 0.2

    my_x_ticks = np.arange(0, 19, 1)
    my_y_ticks = np.arange(0, 0.45, 0.05)
    plt.xticks(my_x_ticks)
    plt.yticks(my_y_ticks)

    plt.bar(x, ratio, width=width, label='TS-W', fc='grey')

    plt.tight_layout()
    plt.show()
    # plt.savefig("../../output/histogram.eps", dpi=600, format='eps')
    # plt.savefig("../../output/histogram.jpg")
    # print("save output/histogram.jpg")

    avg /= 20
    print(avg)


def test_box_plot():
    all_data = [[[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
                [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]],
                [[2, 2, 3, 4, 3, 5, 6], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]]]

    data = [[1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2], [1, 2, 3, 4, 3, 5, 2]]
    pp = BoxPlot()
    # pp.get_box_plot(data)
    pp.get_box_plot_multi_groups(all_data)


# test_plot()
test_plot_multi_groups()
# test_plot_cactus()
# test_plot_pinpoint()
# test_plot_bar()
# test_box_plot()
