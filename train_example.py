#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import datautils
from fm import FMClassifier, FMRegressor


# model related
INP_DIM = 18765
HID_DIM = 128
REG_W = 0.1
REG_V = 0.1
# training related
LR = 1e-3
MAX_ITER = 100
EVAL_ITER = 2
BATCH_SIZE = 128
# dump related
MDL_CKPT_DIR = './model_ckpt/model.ckpt'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'
LOG_PATH = './tensorboard_log'

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
    lambda_v=REG_V)

# init session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction=0.333
sess = tf.Session(config=config)

# init tensorboard writer
train_writer = tf.summary.FileWriter(LOG_PATH + '/train', sess.graph)
test_writer = tf.summary.FileWriter(LOG_PATH + '/test')

# init variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

niter = 0
while niter < MAX_ITER:
    niter += 1
    batch_data = freader.get_batch(BATCH_SIZE)
    if not batch_data:
        break
    train_x, train_y = inp_fn(batch_data, INP_DIM)
    train_summary_loss, train_loss, _ = mdl.train_step(sess, train_x, train_y, lr=LR)
    train_writer.add_summary(train_summary_loss, niter)
    if niter % EVAL_ITER == 0:
        # each metric could be evaluated separately
        # test_summary_loss, test_loss = mdl.eval_loss(sess, test_x, test_y)
        # test_summary_auc, test_auc = mdl.eval_auc(sess, test_x, test_y)
        # test_writer.add_summary(test_summary_loss, niter)
        # test_writer.add_summary(test_summary_auc, niter)
        test_summary, test_loss, test_auc = mdl.eval_metrics(sess, test_x, test_y)
        test_writer.add_summary(test_summary, niter)
    else:
        test_loss = '-----'
        test_auc = '-----'
    print(niter, 'train:', train_loss, 'test_loss:', test_loss, 'test_auc:', test_auc)
save_path = mdl.saver.save(sess, MDL_CKPT_DIR, global_step=mdl.global_step)
print('model saved:', save_path)

sess.close()
