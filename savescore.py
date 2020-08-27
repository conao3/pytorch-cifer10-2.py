import numpy as np
import matplotlib.pyplot as plt


def data_check(data, length):
    datalength = len(data)

    if datalength == 1:
        a = np.zeros(length)
        a[:] = data[0]
        data = a
    else:
        for i in range(datalength):
            l_i = len(data[i])
            if length != l_i:
                a = np.zeros(length)
                a[:l_i] = data[i]
                a[l_i:] = data[i][-1]
                data[i] = a
    return data


def epand_plot_score(x_count,
                     data,
                     x_lim=None,
                     y_lim=None,
                     x_ticks=None,
                     x_label='EPOCH',
                     y_label='score',
                     title='score',
                     legend=['train', 'test'],
                     filename='test'):
    plt.figure(figsize=(6, 6))

    if x_ticks is None:
        x_ticks = [
            '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9',
            '1.0'
        ]

    datalength = len(data)

    data = data_check(data, x_count)

    if x_lim is None:
        x_lim = [0, x_count]
    if y_lim is None:
        y_lim = [0, 1]

    for i in range(datalength):
        plt.plot(range(x_count), data[i])

    plt.ylim(y_lim[0], y_lim[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(np.arange(x_count + 1), x_ticks)
    plt.legend(legend)
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.close()


def plot_score(epoch,
               train_data,
               test_data,
               x_lim=None,
               y_lim=None,
               x_label='EPOCH',
               y_label='score',
               title='score',
               legend=['train', 'test'],
               filename='test'):
    plt.figure(figsize=(6, 6))

    if x_lim is None:
        x_lim = epoch
    if y_lim is None:
        y_lim = 1

    plt.plot(range(epoch), train_data)
    plt.plot(range(epoch), test_data, c='#00ff00')
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(legend)
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.close()


def save_data(train_loss, test_loss, train_acc, test_acc, filename):
    with open(filename + '.txt', mode='w') as f:
        f.write("train mean loss={}\n".format(train_loss[-1]))
        f.write("test  mean loss={}\n".format(test_loss[-1]))
        f.write("train accuracy={}\n".format(train_acc[-1]))
        f.write("test  accuracy={}\n".format(test_acc[-1]))


def plot_one(epoch,
             data,
             x_lim=None,
             y_lim=None,
             x_label='EPOCH',
             y_label='score',
             title='score',
             filename='test'):
    plt.figure(figsize=(6, 6))

    if x_lim is None:
        x_lim = epoch
    if y_lim is None:
        y_lim = 1

    plt.plot(range(epoch), data)
    plt.xlim(0, x_lim)
    plt.ylim(0, y_lim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.close()
