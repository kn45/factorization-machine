#!/usr/bin/env python


import numpy as np
import sys
import tensorflow as tf
import datautils
from fm import FMRegressor

INP_DIM = 18765
HID_DIM = 128
REG_W = 0.1
REG_V = 0.1

LR = 1e-4
TOTAL_ITER = 100

MDL_CKPT_DIR = './model_ckpt/model.ckpt'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'


inp_fn = datautils.idx_inp_fn
# inp_fn = datautils.libsvm_inp_fn

freader = datautils.BatchReader(TRAIN_FILE)
with open(TEST_FILE) as ftest:
    test_data = [x.rstrip('\n') for x in ftest.readlines()]
test_x, test_y = inp_fn(test_data, INP_DIM)

mdl = FMRegressor(
    inp_dim=INP_DIM,
    hid_dim=HID_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V,
    lr=LR)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
niter = 0

while niter < TOTAL_ITER:
    niter += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data, INP_DIM)
    mdl.train_step(sess, train_x, train_y)
    train_eval = mdl.eval_step(sess, train_x, train_y)
    test_eval = mdl.eval_step(sess, test_x, test_y) \
        if niter % 1 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
print "model saved:", save_path

with open('train_done_test_res', 'w') as f:
    preds = mdl.predict(sess, test_x)
    for l, p in zip(test_y, preds):
        print >> f, '\t'.join(map(str, [l[0], p[0]]))
    embs = mdl.get_embedding(sess, test_x)
    for e in embs:
        print >> f, e

sess.close()
