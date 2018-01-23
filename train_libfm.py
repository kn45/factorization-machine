#!/usr/bin/env python


import sys
import tensorflow as tf
sys.path.append('../utils')
import dataproc
from fm import FMClassifier

INP_DIM = 310624
HID_DIM = 256
REG_W = 0.0  # 1st
REG_V = 0.0  # 2nd

LR = 1e-3
TOTAL_ITER = 400000

MDL_CKPT_DIR = './model_ckpt/model.ckpt'
TRAIN_FILE = '../data/train_feature.libfm'
# TEST_FILE = '../data/test_feature.libfm'
TEST_FILE = '../data/test_feature.libfm.samp'


def inp_fn(data):
    bs = len(data)
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split(' ')
        label = float(flds[0])
        feats = flds[1:]
        for feat in feats:
            idx, val = feat.split(':')
            idx = int(idx)
            val = float(val)
            x_idx.append([i, idx])
            x_vals.append(val)
        y_vals.append([label])
    x_shape = [bs, INP_DIM]
    return (x_idx, x_vals, x_shape), y_vals


freader = dataproc.BatchReader(TRAIN_FILE)
with open(TEST_FILE) as ftest:
    test_data = [x.rstrip('\n') for x in ftest.readlines()]
test_x, test_y = inp_fn(test_data)

mdl = FMClassifier(
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
    batch_data = freader.get_batch(256)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data)
    mdl.train_step(sess, train_x, train_y)
    train_eval = mdl.eval_step(sess, train_x, train_y)
    test_eval = mdl.eval_step(sess, test_x, test_y) \
        if niter % 50 == 0 else 'SKIP'
    print niter, 'train:', train_eval, 'test:', test_eval
    if niter % 2000 == 0:
        save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
print "model saved:", save_path

sess.close()
