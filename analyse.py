import numpy as np
import glob as g
import matplotlib.pyplot as plt


def read_durations(path):
    f = open('./log/' + path, 'r')
    res = []

    for line in f.readlines():
        res.append(float(line))

    return np.array(res)


def percentiles(a):
    return np.percentile(a, 25), np.percentile(a, 50), np.percentile(a, 75)


def read_positions(s_path):
    l = sorted(g.glob('./log/' + s_path))
    res = []

    for path in l:
        f = open(path, 'r')
        tres = []

        for line in f.readlines():
            tline = line.split()
            x = float(tline[0])
            y = float(tline[1])
            tres.append([x, y])

        res.append(tres)

    return res


def first_positions(l):
    res = []

    for i in range(10):
        res += l[i]

    return np.array(res)


def last_positions(l):
    res = []

    for i in range(10):
        res += l[len(l) - 1 - i]

    return np.array(res)


def draw_hist(csv):
    data = np.genfromtxt(csv, delimiter=',')
    x = data[:, 0]
    y = data[:, 1]
    plt.hist2d(x, y, bins=50, range=[[0, 600], [0, 600]])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(csv)
    plt.show(block=False)
    plt.savefig(csv[:-3] + "png")
    plt.close()


def read_npy(path):
    res = np.load('./log/' + path, allow_pickle=True)

    return res


def draw_durations(s_path):
    l = sorted(g.glob('./log/' + s_path))
    res = []

    for path in l:
        f = open(path, 'r')
        tres = []

        for line in f.readlines():
            tline = line.split()
            t = float(tline[0])
            tres.append(t)

        res.append(tres)

    runs = np.array([i for i in range(1, len(res[0]) + 1)])

    for i, rl in enumerate(res):
        plt.plot(runs, rl, label='launch ' + str(i + 1))

    plt.xlabel('Run')
    plt.ylabel('Time (s)')
    plt.title('Durations')
    plt.grid()
    plt.legend()
    plt.show(block=False)
    plt.savefig('Durations.png')
    plt.close()
    print('Durations.png est sauvé.')

    return res


def stats_durations(l):
    fd = []
    ld = []

    for durations in l:
        fd += durations[:10]
        ld += durations[-10:]

    print('first runs : q1 :', np.percentile(fd, 25), 'm :', np.percentile(fd, 50), 'q3 :', np.percentile(fd, 75))
    print('last runs : q1 :', np.percentile(ld, 25), 'm :', np.percentile(ld, 50), 'q3 :', np.percentile(ld, 75))

    return


if __name__ == '__main__':
    a = read_durations('1641418414.6273637-TrialDurations-randomPersist.txt')
    # print('1641418414.6273637-TrialDurations-randomPersist.txt :')
    # q1, m, q3 = percentiles(a)
    # print('q1 :', q1, 'm :', m, 'q3 :', q3)
    # b = read_positions("Thu Jan  6 08:41:16 2022*-Trial-*")
    # fp = first_positions(b)
    # lp = last_positions(b)
    # np.savetxt('histoDebut.csv', fp, delimiter=',')
    # np.savetxt('histoFin.csv', lp, delimiter=',')
    # draw_hist('histoDebut.csv')
    # draw_hist('histoFin.csv')
    # print('histoDebut.csv, histoFin.csv et leurs histo. sont sauvés.')
    # q = read_npy('Thu Jan  6 08:41:16 2022-TrialQvalues-qlearning.npy')
    # keys = ['00002', '00072', '00000', '00070', '11101', '11171']
    # q = q.flatten()[0]
    #
    # for key in keys:
    #     print(key, ':', q[key])
    #
    # d = draw_durations('*TrialDurations-qlearning*')
    # stats_durations(d)
    #

    print(percentiles(a))





