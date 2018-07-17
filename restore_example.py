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
# dump related
MDL_CKPT_DIR = './model_ckpt/'
TRAIN_FILE = './rt-polarity.shuf.train'
TEST_FILE = './rt-polarity.shuf.test'
# feed function related
feed_fn = datautils.idx_inp_fn
# feed_fn = datautils.libsvm_inp_fn

# read test data
test_x, test_y = feed_fn([x.rstrip('\n') for x in open(TEST_FILE).readlines()], INPUT_DIM)

mdl = FMClassifier(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    lambda_w=REG_W,
    lambda_v=REG_V)

sess = tf.Session()
mdl.saver.restore(sess, tf.train.latest_checkpoint(MDL_CKPT_DIR))
print('Global steps:', sess.run(mdl.global_step))

with open('train_done_test_res', 'w') as f:
    preds = mdl.predict_proba(sess, test_x)
    for l, p in zip(test_y, preds):
        print(*map(str, [l[0], p[0]]), sep='\t', file=f)
    embs = mdl.get_embedding(sess, test_x)
    for e in embs:
        print(e, file=f)

sess.close()
