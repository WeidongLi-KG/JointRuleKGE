# -*- coding: utf-8 -*-
"""
# @Created on: 2018/5/29 16:07
# @Author: WeidongLi

"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

mpl.rcParams['font.sans-serif'] = ['SimHei']
def read_file_data(filename="result_01.txt"):
    result = []
    with open(filename) as f:
        for line in f:
            if line[:4]=='2018':
                loss = line.strip().split()
                result.append(float(loss[-1]))
                # print(result)
    return np.asarray(result)

if __name__ == '__main__':
    loss1 = read_file_data()
    loss2 = read_file_data('result_02.txt')
    x = np.linspace(1000,100000,100,dtype=float)
    fig = plt.figure()

    # ax = plt.gca()
    fig.subplots_adjust(left=2.0,right=2.1,hspace=1.2,wspace=1.0)
    print(repr(x),repr(loss1),sep='\n')
    print(len(loss1),len(loss2))
    plt.figure()
    ax = plt.subplot(111)
    line1 = ax.plot(x,loss2,'m-')
    line2 = ax.plot(x,loss1,'c-')
    plt.title('hehe')
    plt.xlabel('训练步数')
    plt.ylabel('损失函数损失值')
    a = np.arange(0,100001,20000)
    print(len(a))
    b = np.linspace(0,5,6)*0.1
    print(b)

    plt.yticks(np.arange(0.0,.51,0.1))

    # plt.xticks([])
    # fmt = FormatStrFormatter('%d')
    # ax.xaxis.set_minor_formatter(fmt)
    # ax.set_xticks(np.arange(0,100001,20000),minor=True)
    # ax.set_xticks(np.arange(1,100000,10000),minor=True)
    # ax.set_xticklabels([])
    # ax.set_xticklabels(np.arange(0,100000,10000))
    # plt.yticks()
    ax.set_xticks(np.arange(0,100001,10000),minor=True)
    ax.set_xticklabels([],minor=False)
    ax.set_xticklabels(np.arange(0,100001,10000),minor=True)
    ax.set_yticks(np.arange(0.0,0.5,0.05),minor=True)
    # ax.set_yticklabels([0.05,"",0.15],minor=True)
    # plt.xticks(a)
    # plt.ylim(0.0,0.5)
    # plt.ylim(0.0,0.5)


    plt.show()


#
# import numpy as np
# import matplotlib.pyplot as plt
#
# # 0到5之间每隔0.2取一个数
# t = np.arange(0., 5., 0.2)
#
# # 红色的破折号，蓝色的方块，绿色的三角形
# plt.plot(t, t, 'r--', t, t**3, 'g^')
# plt.plot(t,t**2)
# plt.show()