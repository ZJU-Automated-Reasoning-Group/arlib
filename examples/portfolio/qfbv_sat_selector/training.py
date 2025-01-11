import torch
import pandas as pd
import numpy as np
from torch.utils import data
from sklearn.utils import shuffle
from IPython import display
from torch import nn
from d2l import torch as d2l
from collections import Counter

# df = pd.read_csv('dataset.csv' , sep=',',header=None)
# print(df.to_string())
sat_solvers_in_pysat = ['gc3', 'gc4', 'g3',
                        'g4', 'mcb', 'mpl', 'mg3',
                        'mc', 'm22', 'mgh']


def analyze_data():
    df = pd.read_csv('dataset.csv', sep=',', header=None)
    features, labels = df.iloc[:, 0:38], df.iloc[:, 38:39]
    labels = labels.values.reshape((-1,))
    label_count = pd.value_counts(labels)
    # print(label_count)
    print(f"Total {len(labels)} cnf")
    for idx, value in label_count.items():
        if idx < 0:
            print(f"{value} cnf unsolvable")
        else:
            print(f"{sat_solvers_in_pysat[idx]} can solve {value} cnf files")


def synthetic_data():
    df = pd.read_csv('dataset.csv', sep=',', header=None)
    df = shuffle(df)
    sz = len(df)
    features, labels = df.iloc[:, 0:38], df.iloc[:, 38:39]
    np.reshape(labels.values, (sz,))
    # 9th solver works best
    to_delete = []
    # delete dfeaturesata that all solvers can not solve
    for i in range(sz):
        if labels.values[i, 0] == -1:
            to_delete.append(i)
    features = features.values[:]
    labels = labels.values[:]
    features = np.delete(features, to_delete, axis=0)
    labels = np.delete(labels, to_delete, axis=0)
    # shuffle features and labels
    state = np.random.get_state()
    np.random.shuffle(features)
    np.random.set_state(state)
    np.random.shuffle(labels)
    features = torch.tensor(features)
    labels = torch.tensor(labels)
    labels = labels.to(torch.float32)
    features = features.to(torch.float32)
    sz_train = int(len(features) * 4 / 5)
    train_features = features[:sz_train, :]
    train_labels = labels[:sz_train, :]
    test_features = features[sz_train:, :]
    test_labels = labels[sz_train:, :]
    return train_features, train_labels, test_features, test_labels


# 返回迭代器
def load_array(data_arrays, batch_size, is_train=True):  # @save
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:  # @save
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def train_epoch_ch3(net, train_iter, loss, updater):
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
        net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator:  # @save
    """在动画中绘制数据"""

    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    # assert train_loss < 0.5, train_loss
    # assert train_acc <= 1 and train_acc > 0.7, train_acc
    # assert test_acc <= 1 and test_acc > 0.7, test_acc
    d2l.plt.show()


batch_size = 10


def train_main():
    train_features, train_labels, test_features, test_labels = synthetic_data()
    train_iter = load_array((train_features, train_labels), batch_size)
    test_iter = load_array((test_features, test_labels), batch_size)
    # PyTorch不会隐式地调整输入的形状。因此，
    # 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
    net = nn.Sequential(nn.Flatten(), nn.Linear(38, 10))

    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net.apply(init_weights)
    loss = nn.MultiLabelSoftMarginLoss(reduction='none')
    # loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)
    num_epochs = 30
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)


if __name__ == "__main__":
    analyze_data()
