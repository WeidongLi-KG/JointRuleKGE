# -*- coding: utf-8 -*-
# @Time    : 2018/6/15 17:22
# @Author  : WeiDongLi
# @File    : plot_res_eacc_pacc.py

import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['font.style']=['italic']
step,loss,pacc,eacc = list(),list(),list(),list()
with open('./result.txt','r') as res_file:
    for line in res_file:
        res = line.strip().split()
        n_step = res[2].strip(',')
        step.append(int(n_step))
        loss.append(float(res[4]))
        pacc.append(float(res[6]))
        eacc.append(float(res[8]))
# print(step)
plt.subplot(212)
plt.title("实体分类准确率曲线")
plt.plot(step,pacc,'b-',step,eacc,'c-')
plt.legend(['pacc','eacc'])
plt.xlabel('训练步数')
plt.ylabel('pacc/eacc')
plt.annotate("exact accuracy",xy = (800,0.4),xytext=(600,0.1),color='k',fontstyle='italic',
             arrowprops=dict(facecolor='cyan',edgecolor='c',arrowstyle='->',),fontsize='16')
plt.annotate("partial accuracy",xy = (400,0.5),xytext=(150,0.1),color = 'b',
             arrowprops=dict(facecolor='k',edgecolor='b',arrowstyle='->',),fontsize='16',)

plt.subplot(211)
the_time = datetime.now().isoformat(sep='-')
plt.text(500,3.0,the_time,fontdict=dict(fontsize='large'))
plt.ylabel("softmax 交叉熵损失值")
plt.xlabel("训练步数")
plt.plot(step,loss,'r-')
plt.legend(["损失曲线"])
plt.text(180,4.0,"WeidongLi",fontdict=dict(fontsize=14,fontstyle='italic' ),fontstyle='italic',rotation=20)
plt.annotate('损失函数曲线', xy=(300, 2.5), xytext=(400, 4.0),arrowprops=dict(facecolor='magenta', width=5,
                                                                        shrink=0.001))
plt.title("神经网络实体分类损失函数优化曲线",fontstyle='italic')
plt.subplots_adjust(hspace=0.5)
plt.show()

list