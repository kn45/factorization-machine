#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
import datautils
from fm import FMClassifier, FMRegressor


# model related
INPUT_DIM = 18765
HIDDEN_DIM = 128
REG_W = 0.0
REG_V = 0.0
# training related
LEARNING_RATE = 1e-3
MAX_ITER = 200
EVAL_ITER = 2
BATCH_SIZE = 128
# dump related
MDL_CKPT_DIR = './model_ckpt/model.ckpt'
MDL_CMPT_DIR = './model_cmpt/model.ckpt'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'
LOG_PATH = './tensorboard_log'
# feed function related
feed_fn = datautils.index_input_func

# train data and test data
train_reader = datautils.BatchReader(TRAIN_FILE, batch_size=BATCH_SIZE)
test_x, test_y = feed_fn([x.rstrip('\n') for x in open(TEST_FILE).readlines()], INPUT_DIM)

# define model
model_ = FMClassifier(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V)

# init session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.333
sess = tf.Session(config=config)

# init tensorboard writer
train_writer = tf.summary.FileWriter(LOG_PATH + '/train', sess.graph)
test_writer = tf.summary.FileWriter(LOG_PATH + '/test')

# init variables
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

for niter, batch_data in enumerate(train_reader):
    if niter >= MAX_ITER:
        break
    train_x, train_y = feed_fn(batch_data, INPUT_DIM)
    train_summary_loss, train_loss, _ = model_.train_step(sess, train_x, train_y, lr=LEARNING_RATE)
    train_writer.add_summary(train_summary_loss, niter)
    if niter % EVAL_ITER == 0:
        # each metric could be evaluated separately
        # test_summary_loss, test_loss = model_.eval_loss(sess, test_x, test_y)
        # test_summary_auc, test_auc = model_.eval_auc(sess, test_x, test_y)
        # test_writer.add_summary(test_summary_loss, niter)
        # test_writer.add_summary(test_summary_auc, niter)
        test_summary, test_loss, test_auc = model_.eval_metrics(sess, test_x, test_y)
        test_writer.add_summary(test_summary, niter)
    else:
        test_loss = '-----'
        test_auc = '-----'
    print(niter, 'train:', train_loss, 'test_loss:', test_loss, 'test_auc:', test_auc)
save_path = model_.ckpt_saver.save(sess, MDL_CKPT_DIR, global_step=model_.global_step)
save_path = model_.saver.save(sess, MDL_CMPT_DIR, global_step=model_.global_step)
print('model saved:', save_path)

sess.close()
