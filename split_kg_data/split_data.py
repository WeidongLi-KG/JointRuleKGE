#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2018/10/10 10:52
# @Author  : WeidongLi
# @Email   : weidonghappy@163.com
# @File    : split_data.py
# @Software: PyCharm
import argparse
import numpy as np
import os
import sys
import logging
from tqdm import tqdm

parser = argparse.ArgumentParser(description=sys.argv[0])
parser.add_argument('--input_dir', '-i', type=str, default='../data/FB15k/', help='Input data directory')
parser.add_argument('--output_dir', '-o', type=str, default='../data/FB15k/output/', help='Output data directory')
parser.add_argument('--train', '-n', type=int, default='10000', help='Output train triple size')
parser.add_argument('--valid_size', '-v', type=int, default='5000', help='Valid triple size')
parser.add_argument('--test_size', '-t', type=int, default='5000', help='test triple size')
args = parser.parse_args()
print(args)
logging.basicConfig(level=logging.INFO)

def split_dataset(input_dir, output_dir, args):
    with open(input_dir+'train.txt', 'r') as f:
        data = np.asarray([line.strip().split('\t') for line in f])
        train_data = np.random.permutation(data)[:args.train]
        logging.info('\ntrain data shape : {}'.format(train_data.shape))
        train_size = len(train_data)
        if not os.path.exists(args.output_dir): os.makedirs(args.output_dir)
        with open(args.output_dir+'train.txt', 'w') as fw:
            fw.writelines([train_data[i, 0]+'\t'+train_data[i, 1]+'\t'+train_data[i, 2] + '\n' for i in tqdm(range(train_size))])

        train_ent, train_rel = set(), set()
        ent_dict, rel_dict = dict(), dict()
        logging.info('\ngenerate entity and relation set')
        for i in tqdm(range(train_size)):
            train_ent.add(train_data[i, 0])
            train_ent.add(train_data[i, 1])
            train_rel.add(train_data[i, 2])


        logging.info('\nTranslating set to list, because set object can not be indexed\n')
        ent_list = list(train_ent); rel_list = list(train_rel)
        entity2id_f = open(args.output_dir+'entity2id.txt', 'w')
        relation2id_f = open(args.output_dir+'relation2id.txt', 'w')

        logging.info('\nWriting entity2id file'+'-'*10)
        for i in tqdm(range(len(train_ent))):
            ent_dict[ent_list[i]] = i
            entity2id_f.write(ent_list[i]+'\t'+str(i)+'\n')
        entity2id_f.close()

        logging.info('\nWriting relation2id file'+'-'*10)
        for i in tqdm(range(len(train_rel))):
            rel_dict[rel_list[i]] = i
            relation2id_f.write(rel_list[i]+'\t'+str(i)+'\n')
        relation2id_f.close()

        valid_f = open(input_dir+'valid.txt', 'r')
        test_f = open(input_dir+'test.txt', 'r')
        valid_fo = open(output_dir+'valid.txt', 'w')
        test_fo = open(output_dir+'test.txt', 'w')

        valid_data = np.asarray([line.strip().split('\t') for line in valid_f])
        logging.info('\nvalid_data shape: {}'.format(valid_data.shape))
        test_data = np.asarray([line.strip().split('\t') for line in test_f])
        logging.info('\ntest_data shape: {}'.format(test_data.shape))
        valid_f.close();test_f.close()

        valid_triple_num = 0
        for i in tqdm(range(len(valid_data))):
            h, r, t = valid_data[i, 0], valid_data[i, 2], valid_data[i, 1]
            if (h in ent_dict) and (t in ent_dict) and (r in rel_dict):
                valid_fo.write(h+'\t'+t+'\t'+r+'\n')
                valid_triple_num += 1
                if valid_triple_num == args.valid_size: break
            else: continue
        logging.info('\nHave written data to valid.txt, the valid triple number: {}'.format(valid_triple_num))

        test_triple_num = 0
        for i in tqdm(range(len(test_data))):
            h, r, t = test_data[i, 0], test_data[i, 2], test_data[i, 1]
            if (h in ent_dict) and (t in ent_dict) and (r in rel_dict):
                test_fo.write(h+'\t'+t+'\t'+r+'\n')
                test_triple_num += 1
                if test_triple_num == args.test_size: break
            else: continue

        logging.info('\nHave written data to test.txt, the test triple number: {}'.format(test_triple_num))
        valid_fo.close();test_fo.close()

if __name__ == '__main__':
    split_dataset(args.input_dir, args.output_dir, args)


















