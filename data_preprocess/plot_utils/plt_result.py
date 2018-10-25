# -*- coding: utf-8 -*-
"""
# @Created on: 2018/5/29 9:30
# @Author: WeidongLi

"""
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib as mpl
import numpy as np
#%matplotlab inline
mpl.rcParams['font.sans-serif']=['SimHei']
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
    print(len(loss1),len(loss2))
    x = np.linspace(1000, 100000, 100, dtype=np.int32)
    xmajorLocator = MultipleLocator(20000)
    xmajorFormatter = FormatStrFormatter('%d')
    xminorLocator = MultipleLocator(10000)
    ymajorLocator = MultipleLocator(0.05)
    ymajorFormater = FormatStrFormatter('%7.5f')
    yminorLocator = MultipleLocator(0.025)
    fig = plt.figure(1,figsize=(50,10))
    # fig.figure(figsize=(5, 1))
    # fig.subplots_adjust(left=0.5,right=1.5)

    ax = fig.add_subplot(111)

    str_id = '''
# @Created on: 2018/6/19 9:30
# @Author: WeidongLi'''

    line2d = ax.plot(x,loss1,'b-',x,loss2,'r-')
    print(line2d)
    # plt.ylim((0.0001,0.5),(0.0001,0.5))
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)

    ax.yaxis.grid(which='major',color='c',linestyle='--')
    # ax.set_xticklabels([])
    xticks = np.arange(0,100001,20000,dtype=int)
    # ax.set_xticklabels(xticks)
    # ax.xaxis.set_ticklabels(xticks)
    # ax.xaxis.grid()
    print(repr(xticks))
    ax.yaxis.set_major_formatter(ymajorFormater)
    ax.yaxis.set_major_locator(ymajorLocator)

    ax.yaxis.set_minor_locator(yminorLocator)
    ax.set_title("知识库补全算法实验(不同学习率下的损失函数收敛效果)",fontdict=dict(fontsize=30))
    ax.set_xlabel("训练次数（十万次）",fontdict=dict(fontsize=20))
    ax.set_ylabel("损失函数损失值",fontdict=dict(fontsize=20))
    ax.text(6000,0.265,'acc：Hits@3 0.333 Hits@10 0.508',fontsize=24,color='r',rotation=15)
    ax.text(20000,0.065,'acc：Hits@3 0.105 Hits@10 0.184',fontsize=24,color='b')
    ax.legend(line2d,['学习率：0.01    ','学习率：0.0001    '],fontsize=24)
    ax.text(20000,0.41,str_id,fontsize=22,color='m')
    ax.set_ylim(-0.01, 0.5)
    ax.set_xlim(-1000, 100000)
    # fig.savefig('result.eps')
    plt.show()
    fig.savefig('result.jpeg')




