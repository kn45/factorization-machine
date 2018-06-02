#!/usr/bin/env python

import numpy as np
import sys
import tensorflow as tf
import datautils
from fm import FMClassifier


# model related
INP_DIM = 18765
HID_DIM = 128
REG_W = 0.1
REG_V = 0.1
# training related
LR = 1e-4
MAX_ITER = 100
EVAL_ITER = 2
# dump related
MDL_CKPT_DIR = './model_ckpt/model.ckpt'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'
LOG_PATH = './train_log'

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
file_writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
niter = 0

while niter < MAX_ITER:
    niter += 1
    batch_data = freader.get_batch(128)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data, INP_DIM)
    mdl.train_step(sess, train_x, train_y)
    train_loss = mdl.eval_loss(sess, train_x, train_y)
    if niter % EVAL_ITER == 0:
        test_loss = mdl.eval_loss(sess, test_x, test_y)
        test_auc = mdl.eval_auc(sess, test_x, test_y)
    else:
        test_loss = '-----'
        test_auc = '-----'
    print niter, 'train:', train_loss, 'test_loss:', test_loss, 'test_auc:', test_auc
save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
print "model saved:", save_path

sess.close()
