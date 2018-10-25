# -*- coding: utf-8 -*-
# @Time    : 2018/9/28 17:35
# @Author  : WeidongLi
# @Email   : weidonghappy@163.com
# @File    : gather_tf_ops_sp.py
# @Software: PyCharm
import tensorflow as tf
tf.reset_default_graph()

# temp = tf.range(0,10)*10 + tf.constant(1,shape=[10,10])
# temp2 = tf.gather(temp,[[1,2]])
#
# with tf.Session() as sess:
#
#     print (sess.run(temp))
#     print (sess.run(temp2))
E = tf.Variable(tf.ones((10,1)),) #d=embed size shape=(67447,100)
W = [tf.Variable(tf.truncated_normal([5,5,3]),) for r in range(8)]
V = [tf.Variable(tf.zeros([3, 2*5]),) for r in range(8)]
b = [tf.Variable(tf.zeros([3, 1]),) for r in range(8)]

num_rel_r = tf.expand_dims(10, 0)
print('E_shape',E.shape)
e = tf.reshape(E,num_rel_r)
print('E',E.shape,'\ne',e.shape)
print(num_rel_r.get_shape())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # for var in tf.trainable_variables():
    #     print(var.name,var.shape,var)
    #     print('\n')

    print(sess.run(num_rel_r))
    print('E',sess.run(E),'\ne',sess.run(e))
