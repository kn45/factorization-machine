#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import datautils
from fm import FMClassifier

INP_DIM = 18765
HID_DIM = 128
REG_W = 0.1
REG_V = 0.1

LR = 1e-4
TOTAL_ITER = 100

MDL_DIR = './model_ckpt/'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'


inp_fn = datautils.idx_inp_fn
# inp_fn = datautils.libsvm_inp_fn

freader = datautils.BatchReader(TRAIN_FILE)
with open(TEST_FILE) as ftest:
    test_data = [x.rstrip('\n') for x in ftest.readlines()]
test_x, test_y = inp_fn(test_data, INP_DIM)

mdl = FMClassifier(
    inp_dim=INP_DIM,
    hid_dim=HID_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V,
    lr=LR)

sess = tf.Session()
mdl.saver.restore(sess, tf.train.latest_checkpoint(MDL_DIR))
print('Global steps:', sess.run(mdl.global_step))

with open('train_done_test_res', 'w') as f:
    preds = mdl.predict_proba(sess, test_x)
    for l, p in zip(test_y, preds):
        print(*map(str, [l[0], p[0]]), sep='\t', file=f)
    embs = mdl.get_embedding(sess, test_x)
    for e in embs:
        print(e, file=f)

sess.close()
