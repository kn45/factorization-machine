# -*- coding=utf-8 -*-
import copy
import numpy as np
import sys
from operator import itemgetter


class BatchReader(object):
    """Get batch data recurrently from a file.
    """
    def __init__(self, file_name, max_epoch=None):
        self.fname = file_name
        self.max_epoch = max_epoch
        self.nepoch = 0
        self.fp = None

    def __del__(self):
        if self.fp:
            self.fp.close()

    def get_batch(self, batch_size, out=None):
        if out is None:
            out = []
        if not self.fp:
            if (not self.max_epoch) or self.nepoch < self.max_epoch:
                # if max_epoch not set or num_epoch not reach the limit
                self.fp = open(self.fname)
                self.nepoch += 1
            else:  # reach max_epoch limit
                return out
        for line in self.fp:
            out.append(line.rstrip('\n'))
            if len(out) >= batch_size:
                break
        else:
            self.fp.close()
            self.fp = None
            return self.get_batch(batch_size, out)
        return out


def idx_inp_fn(data, dim):
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
    x_shape = [bs, dim]
    return (x_idx, x_vals, x_shape), y_vals


def seq_inp_fn(data):
    bs = len(data)
    max_len = 0
    x_idx = []
    x_vals = []
    y_vals = []
    for i, inst in enumerate(data):
        flds = inst.split('\t')
        label = float(flds[0])
        feats = sorted(map(int, flds[1:]))
        if len(feats) > max_len:
            max_len = len(feats)
        for col, feat in enumerate(feats):
            x_idx.append([i, col])
            x_vals.append(feat)
        y_vals.append([label])
    x_shape = [bs, max_len]
    return (x_idx, x_vals, x_shape), y_vals


def libsvm_inp_fn(data, dim):
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
    x_shape = [bs, dim]
    return (x_idx, x_vals, x_shape), y_vals


def draw_progress(iteration, total, pref='Progress:', suff='',
                  decimals=1, barlen=50):
    """Call in a loop to create terminal progress bar
    """
    formatStr = "{0:." + str(decimals) + "f}"
    pcts = formatStr.format(100 * (iteration / float(total)))
    filledlen = int(round(barlen * iteration / float(total)))
    bar = '█' * filledlen + '-' * (barlen - filledlen)
    out_str = '\r%s |%s| %s%s %s' % (pref, bar, pcts, '%', suff)
    out_str = '\x1b[0;34;40m' + out_str + '\x1b[0m'
    sys.stderr.write(out_str),
    if iteration == total:
        sys.stderr.write('\n')
    sys.stderr.flush()
