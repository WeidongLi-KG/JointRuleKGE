#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/10/25 20:20
# @Author  : WeidongLi
# @Email   : weidonghappy@163.com
# @File    : looooggggg.py
# @Software: PyCharm
import logging
import datetime


logging.basicConfig(level=logging.DEBUG, format='%(levelname)s %(module)s line:%(lineno)d %(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
formatter = logging.Formatter('%(levelname)s %(module)s line:%(lineno)d %(asctime)s %(message)s')
filehdlr = logging.FileHandler('log-' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
# datetime.now().strftime('%Y-%m-%d %I:%M:%S.%f %p')
filehdlr.setLevel(logging.DEBUG)
filehdlr.setFormatter(formatter)
logging.getLogger().addHandler(filehdlr)
# filehadler = logging.FileHandler('log-' + datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
# filehadler.setFormatter(formatter)
# filehadler.setLevel(logging.DEBUG)
logging.info('hehe'*100)
