#!/usr/bin/env python


import numpy as np
import sys
import tensorflow as tf
sys.path.append('../MLFlow/utils')
import dataproc
from fm import FMRegressor

INP_DIM = 18765
HID_DIM = 128
REG_W = 0.1
REG_V = 0.1

LR = 1e-4
NEPOCH = 1000

MDL_CKPT_DIR = './model_ckpt/model.ckpt'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'


def inp_fn(data):
    bs = len(data)
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        label = float(flds[0])
        feats = sorted(map(int, flds[1:]))
        for feat in feats:
            x_idx.append([i, feat])
            x_vals.append(1)
        y_vals.append([label])
    x_shape = [bs, INP_DIM]
    return (x_idx, x_vals, x_shape), y_vals


freader = dataproc.BatchReader(TRAIN_FILE)
with open(TEST_FILE) as ftest:
    test_data = [x.rstrip('\n') for x in ftest.readlines()]
test_x, test_y = inp_fn(test_data)

mdl = FMRegressor(
    inp_dim=INP_DIM,
    hid_dim=HID_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V,
    lr=LR)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
epoch = 0

while epoch < NEPOCH:
    epoch += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    train_eval = mdl.eval_step(sess, train_x, train_y)
    test_eval = mdl.eval_step(sess, test_x, test_y) \
        if epoch % 1 == 0 else 'SKIP'
    print epoch, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
print "model saved:", save_path

with open('train_done_test_res', 'w') as f:
    preds = mdl.predict(sess, test_x)
    for l, p in zip(test_y, preds):
        print >> f, '\t'.join(map(str, [l[0], p[1]]))

sess.close()
